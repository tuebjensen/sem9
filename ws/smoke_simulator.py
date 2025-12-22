import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import argparse

print("DEBUG: LD_LIBRARY_PATH: " + os.environ.get("LD_LIBRARY_PATH", "Not Set"))

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from matplotlib import cm
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm, trange


@ti.data_oriented
class FluidSimulator:
    def __init__(self, nx, ny, viscosity, diffusivity, num_sensors=8):
        self.nx = nx
        self.ny = ny

        self.num_sensors = num_sensors

        self.sensor_positions = ti.Vector.field(2, int, shape=num_sensors)
        self.sensor_readings = ti.field(float, shape=(num_sensors, 2))

        self.w = (
            ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        )
        self.e = ti.types.matrix(9, 2, int)(
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]
        )

        self.inv_map = ti.field(int, shape=9)
        inv_data = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
        self.inv_map.from_numpy(inv_data)

        self.viscosity = viscosity
        self.tau_fluid = 3.0 * viscosity + 0.5
        self.inv_tau_fluid = 1.0 / self.tau_fluid

        self.diffusivity = diffusivity
        self.tau_smoke = 3.0 * diffusivity + 0.5
        self.inv_tau_smoke = 1.0 / self.tau_smoke

        self.mask = ti.field(int, shape=(nx, ny))  # 1 for wall, 0 for fluid

        self.rho_fluid = ti.field(float, shape=(nx, ny))
        self.vel_fluid = ti.Vector.field(2, float, shape=(nx, ny))
        self.f_old_fluid = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new_fluid = ti.Vector.field(9, float, shape=(nx, ny))

        self.rho_smoke = ti.field(float, shape=(nx, ny))
        self.g_old_smoke = ti.Vector.field(9, float, shape=(nx, ny))
        self.g_new_smoke = ti.Vector.field(9, float, shape=(nx, ny))

    def set_mask_from_numpy(self, mask_np):
        self.mask.from_numpy(mask_np.astype(np.int32))
        self.enforce_mask_bc()

    def register_sensors(self, sensor_positions):
        n = len(sensor_positions)
        assert n == self.num_sensors, (
            "Number of sensor positions must match num_sensors"
        )

        coords = np.array(sensor_positions, dtype=np.int32)
        self.sensor_positions.from_numpy(coords)

    @ti.kernel
    def read_sensors(self):
        for s in range(self.num_sensors):
            x = self.sensor_positions[s][0]
            y = self.sensor_positions[s][1]
            self.sensor_readings[s, 0] = self.vel_fluid[x, y][0]
            self.sensor_readings[s, 1] = self.vel_fluid[x, y][1]

    def get_sensor_data(self):
        self.read_sensors()
        return self.sensor_readings.to_numpy()

    @ti.kernel
    def enforce_mask_bc(self):
        for i, j in self.mask:
            if self.mask[i, j] == 1:
                # We let the fluid velocity be zero inside the obstacle but let the smoke leak through
                # Looks weird if we set smoke density to zero inside the obstacle
                self.vel_fluid[i, j] = tm.vec2(0.0, 0.0)
                self.rho_smoke[i, j] = 0.0
                self.g_new_smoke[i, j].fill(0.0)
                self.g_old_smoke[i, j].fill(0.0)

    @ti.func
    def f_eq(self, rho, u):
        eu = self.e @ u
        uv = tm.dot(u, u)
        return self.w * rho * (1 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.func
    def g_eq(self, rho, u):
        eu = self.e @ u
        return self.w * rho * (1 + 3.0 * eu)

    @ti.kernel
    def step_fluid(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(self.rho_fluid[ip, jp], self.vel_fluid[ip, jp])
                self.f_new_fluid[i, j][k] = (1 - self.inv_tau_fluid) * self.f_old_fluid[
                    ip, jp
                ][k] + feq[k] * self.inv_tau_fluid

    @ti.kernel
    def step_smoke(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            # If we're inside a wall, zero out smoke
            if self.mask[i, j] == 1:
                self.rho_smoke[i, j] = 0.0
                self.g_new_smoke[i, j].fill(0.0)
                continue

            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]

                incoming_value = 0.0
                rho_eq = 0.0
                vel_eq = tm.vec2(0.0, 0.0)
                # Bounce-back the smoke if the neighboring cell is a wall
                if self.mask[ip, jp] == 1:
                    inv_k = self.inv_map[k]
                    # Use own value for bounce-back
                    incoming_value = self.g_old_smoke[i, j][inv_k]

                    # Pretend the wall has same density/velocity as the fluid
                    # Let the smoke be slightly absorbed by the wall
                    rho_eq = self.rho_smoke[i, j] * 0.985
                    vel_eq = self.vel_fluid[i, j]
                else:
                    # Use neighbor's value if not a wall
                    incoming_value = self.g_old_smoke[ip, jp][k]
                    rho_eq = self.rho_smoke[ip, jp]
                    vel_eq = self.vel_fluid[ip, jp]

                geq = self.g_eq(rho_eq, vel_eq)

                self.g_new_smoke[i, j][k] = (
                    1 - self.inv_tau_smoke
                ) * incoming_value + geq[k] * self.inv_tau_smoke

    @ti.kernel
    def update_macro_vars(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho_fluid[i, j] = 0
            self.vel_fluid[i, j] = 0
            for k in ti.static(range(9)):
                self.f_old_fluid[i, j][k] = self.f_new_fluid[i, j][k]
                self.rho_fluid[i, j] += self.f_new_fluid[i, j][k]
                self.vel_fluid[i, j] += (
                    tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new_fluid[i, j][k]
                )

            if self.rho_fluid[i, j] > 0.0001:
                self.vel_fluid[i, j] /= self.rho_fluid[i, j]
            else:
                self.vel_fluid[i, j] = tm.vec2(0.0, 0.0)

            if self.mask[i, j] == 1:
                self.rho_smoke[i, j] = 0.0
                self.vel_fluid[i, j] = tm.vec2(0.0, 0.0)
                self.g_new_smoke[i, j].fill(0.0)
                self.g_old_smoke[i, j].fill(0.0)
            else:
                self.g_old_smoke[i, j] = self.g_new_smoke[i, j]
                self.rho_smoke[i, j] = self.g_new_smoke[i, j].sum()

    @ti.kernel
    def apply_smoke_source(
        self, cx: float, cy: float, r: float, target_density: float, rate: float
    ):
        for i, j in self.rho_smoke:
            if (i - cx) ** 2 + (j - cy) ** 2 <= r**2:
                self.rho_smoke[i, j] = (
                    rate * target_density + (1.0 - rate) * self.rho_smoke[i, j]
                )
                current_u = self.vel_fluid[i, j]
                new_g_eq = self.g_eq(self.rho_smoke[i, j], current_u)

                self.g_old_smoke[i, j] = new_g_eq
                self.g_new_smoke[i, j] = new_g_eq

    @ti.kernel
    def apply_inlet(self, u_inlet: float):
        for j in range(1, self.ny - 1):
            self.vel_fluid[0, j] = tm.vec2(u_inlet, 0.0)
            self.rho_fluid[0, j] = 1.0

            self.f_old_fluid[0, j] = self.f_eq(1.0, self.vel_fluid[0, j])
            self.f_new_fluid[0, j] = self.f_old_fluid[0, j]

    @ti.kernel
    def apply_outlet(self):
        # Copy the state from the second-to-last column to the last column
        # This allows fluid to leave "smoothly"
        for j in range(1, self.ny - 1):
            # Coordinates for outlet (last col) and neighbor (2nd last col)
            last = self.nx - 1
            prev = self.nx - 2

            # Copy macroscopic variables
            self.rho_fluid[last, j] = self.rho_fluid[prev, j]
            self.vel_fluid[last, j] = self.vel_fluid[prev, j]

            # Copy distribution functions (f) and smoke (g)
            for k in ti.static(range(9)):
                self.f_old_fluid[last, j][k] = self.f_old_fluid[prev, j][k]
                self.f_new_fluid[last, j][k] = self.f_new_fluid[prev, j][k]

                self.g_old_smoke[last, j][k] = self.g_old_smoke[prev, j][k]
                self.g_new_smoke[last, j][k] = self.g_new_smoke[prev, j][k]

            self.rho_smoke[last, j] = self.rho_smoke[prev, j]

    @ti.kernel
    def init_simulation(self):
        self.vel_fluid.fill(0)
        self.rho_fluid.fill(1)
        self.rho_smoke.fill(0)
        self.mask.fill(0)
        for i, j in self.rho_fluid:
            self.f_old_fluid[i, j] = self.f_eq(
                self.rho_fluid[i, j], self.vel_fluid[i, j]
            )
            self.f_new_fluid[i, j] = self.f_old_fluid[i, j]

    @ti.kernel
    def apply_obstacle(self, cx: float, cy: float, r: float):
        for i, j in self.mask:
            if (i - cx) ** 2 + (j - cy) ** 2 <= r**2:
                self.mask[i, j] = 1
                self.vel_fluid[i, j] = tm.vec2(0.0, 0.0)

    def step(self, steps, u_inlet):
        for _ in range(steps):
            self.step_fluid()
            self.step_smoke()
            self.update_macro_vars()
            self.apply_inlet(u_inlet)
            self.apply_outlet()
            self.read_sensors()

    def get_data_snapshot(self):
        smoke_image = self.rho_smoke.to_numpy()
        fluid_velocity = self.vel_fluid.to_numpy()
        return smoke_image, fluid_velocity


def generate_random_inlet_profile(
    timesteps, base_vel=0.1, variation=0.05, smoothness=0.9
):
    profile = []
    current_vel = base_vel
    for t in range(timesteps):
        target = base_vel + variation * np.random.uniform(-1, 1)
        current_vel = (current_vel * smoothness) + (target * (1 - smoothness))
        current_vel = np.clip(current_vel, 0.02, 0.15)
        profile.append(current_vel)

    return np.array(profile)


def generate_simple_sine_inlet_profile(
    timesteps,
    base_vel=0.1,
    amplitude=0.05,
):
    t = np.arange(timesteps)
    p1 = 800

    velocities = np.full(timesteps, base_vel)

    velocities += amplitude * np.sin(2 * np.pi * t / p1)
    velocities = np.clip(velocities, 0.02, 0.15)

    return velocities


def generate_harmonic_inlet_profile(
    timesteps,
    base_vel=0.1,
    amplitude=0.05,
):
    t = np.arange(timesteps)
    p1 = 800
    p2 = 300

    velocities = np.full(timesteps, base_vel)

    velocities += amplitude * np.sin(2 * np.pi * t / p1)
    velocities += (amplitude / 2) * np.sin(2 * np.pi * t / p2)
    velocities += np.random.normal(0, 0.005, timesteps)
    velocities = np.clip(velocities, 0.02, 0.15)

    return velocities


def velocity_field_to_image(vel, sensors, sensor_readings, mask, nx, ny):
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
    vel_magnitude = np.linalg.norm(vel, axis=2).T
    ax.imshow(vel_magnitude, cmap=cm.plasma)

    mask_t = mask.T
    overlay = np.zeros(shape=(*vel_magnitude.shape, 4))
    overlay[mask_t == 1] = [0.5, 0.5, 0.5, 1.0]
    ax.imshow(overlay)

    xs = []
    ys = []
    us = []
    vs = []

    for s in range(len(sensors)):
        xs.append(sensors[s][0])
        ys.append(sensors[s][1])
        us.append(sensor_readings[s][0])
        vs.append(sensor_readings[s][1])

    ax.scatter(xs, ys, c="white", s=10, marker="o", edgecolors="black", zorder=2)
    ax.quiver(
        xs,
        ys,
        us,
        vs,
        scale=0.01,
        width=0.005,
        headwidth=1,
        color="cyan",
        angles="xy",
        scale_units="xy",
        zorder=3,
    )
    ax.set_title("Fluid Velocity Magnitude with Sensor Readings")
    ax.axis("off")

    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    image_data = np.asarray(rgba_buffer)
    plt.close(fig)
    return Image.fromarray(image_data)


def save_gif(frames, save_path, fps=15):
    w, h = frames[0].size
    duration = int(1000 / fps)
    frames[0].save(
        save_path, save_all=True, append_images=frames[1:], duration=duration, loop=0
    )


def generate_biased_mask(nx, ny, smoke_sources, source_safe_zone=40, num_obstacles=3):
    mask = np.zeros((nx, ny), dtype=np.int32)

    target_source = np.random.choice(smoke_sources)
    src_y = target_source["y"]

    safe_zone_start = source_safe_zone
    boundary_margin = 10

    attempts = 0
    obstacles_placed = 0

    while obstacles_placed < num_obstacles and attempts < 1000:
        shape_type = np.random.choice(["circle", "rectangle"])

        if shape_type == "circle":
            r = np.random.randint(8, 10)
            reach_x = r
            reach_y = r
        else:
            w = np.random.randint(15, 30)
            h = np.random.randint(10, 20)
            reach_x = w // 2
            reach_y = h // 2

        min_cx = safe_zone_start + reach_x
        max_cx = nx - boundary_margin - reach_x
        min_cy = boundary_margin + reach_y
        max_cy = ny - boundary_margin - reach_y

        if max_cx <= min_cx or max_cy <= min_cy:
            continue

        cx = int(np.random.triangular(min_cx, min_cx, max_cx))
        if obstacles_placed == 0:
            # Force one obstacle to be in the way of a smoke source
            cy = np.clip(src_y + np.random.randint(-6, 6), min_cy, max_cy)
        else:
            cy = np.random.randint(min_cy, max_cy)

        temp_mask = np.zeros_like(mask)

        if shape_type == "circle":
            y, x = np.ogrid[-cx : nx - cx, -cy : ny - cy]
            index = x * x + y * y <= reach_x * reach_y
            temp_mask[index] = 1

        else:
            # Ensure rectangle is within bounds and not overlapping smoke source
            x_start = cx - reach_x
            x_end = cx + reach_x
            y_start = cy - reach_y
            y_end = cy + reach_y
            temp_mask[x_start:x_end, y_start:y_end] = 1

        overlap = np.logical_and(mask, temp_mask)
        if np.sum(overlap) == 0:
            mask = np.logical_or(mask, temp_mask).astype(np.int32)
            obstacles_placed += 1
        attempts += 1

    return mask


def place_sensors(
    mask,
    num_sensors,
    wall_margin=20,
    obstacle_margin=10,
    sensor_margin=10,
    max_attempts=1000,
):
    dist_map = distance_transform_edt(mask == 0)

    valid_region = dist_map > obstacle_margin
    valid_region[:wall_margin, :] = False
    valid_region[-wall_margin:, :] = False
    valid_region[:, :wall_margin] = False
    valid_region[:, -wall_margin:] = False
    valid_coords = np.argwhere(valid_region)

    def backtrack_solver(current_sensors):
        if len(current_sensors) == num_sensors:
            return True

        for _ in range(max_attempts):
            index = np.random.randint(len(valid_coords))
            rx, ry = valid_coords[index]

            proposal = np.array([rx, ry])
            if current_sensors:
                current_sensors_array = np.array(current_sensors)
                dists = np.linalg.norm(current_sensors_array - proposal, axis=1)
                if np.any(dists < sensor_margin):
                    continue

            current_sensors.append((rx, ry))
            if backtrack_solver(current_sensors):
                return True

            current_sensors.pop()

        return False

    final_sensors = []
    success = backtrack_solver(final_sensors)
    if not success:
        raise RuntimeError("Failed to place sensors with given constraints.")

    return final_sensors


def run_single_simulation(run_id, config, output_root, generate_video=False):
    width = config["width"]
    height = config["height"]
    viscosity = config["viscosity"]
    diffusivity = config["diffusivity"]
    num_sensors = config["num_sensors"]
    total_steps = config["total_steps"]
    snapshot_interval = config["snapshot_interval"]
    seed = config["seed"]
    inlet_profile_type = config["inlet_profile_type"]

    np.random.seed(seed + run_id)

    run_dir = os.path.join(output_root, f"run_{run_id:04d}")
    raw_data_dir = os.path.join(run_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    sim = FluidSimulator(
        nx=width,
        ny=height,
        viscosity=viscosity,
        diffusivity=diffusivity,
        num_sensors=num_sensors,
    )
    sim.init_simulation()

    if inlet_profile_type == "constant":
        wind_profile = np.full(
            total_steps, np.random.uniform(0.08, 0.12)
        ) + np.random.uniform(-0.005, 0.005, total_steps)
    elif inlet_profile_type == "simple_sine":
        wind_profile = generate_simple_sine_inlet_profile(
            total_steps, base_vel=np.random.uniform(0.08, 0.12), amplitude=0.05
        )
    elif inlet_profile_type == "complex_sine":
        wind_profile = generate_harmonic_inlet_profile(
            total_steps, base_vel=np.random.uniform(0.08, 0.12), amplitude=0.05
        )

    smoke_sources = []
    num_smoke_sources = np.random.randint(1, 3)
    for i in range(num_smoke_sources):
        x = np.random.randint(10, 20)
        y = height // (num_smoke_sources + 1) * (i + 1) + np.random.randint(-10, 10)
        r = np.random.randint(4, 8)
        target_density = 1.0
        rate = 0.15
        smoke_sources.append(
            {"x": x, "y": y, "r": r, "target_density": target_density, "rate": rate}
        )

    mask = generate_biased_mask(
        width, height, smoke_sources, num_obstacles=np.random.randint(4, 6)
    )
    sim.set_mask_from_numpy(mask)

    sensor_positions = place_sensors(mask, num_sensors)
    sim.register_sensors(sensor_positions)
    sensor_readings_history = []
    frames_smoke = []
    frames_velocity = []

    sensor_sum = np.zeros(2, dtype=np.float64)
    sensor_sum_sq = np.zeros(2, dtype=np.float64)
    sensor_count = 0

    smoke_sum = 0
    smoke_sum_sq = 0
    smoke_count = 0

    max_expected_velocity = 0.3

    for t in trange(total_steps, desc=f"Simulation Run {run_id:04d}", leave=False):
        sim.step(1, u_inlet=wind_profile[t])

        for i, source in enumerate(smoke_sources):
            # pulse = np.sin(t * 0.005 + i) * 0.2
            sim.apply_smoke_source(
                cx=source["x"],
                cy=source["y"],
                r=source["r"],
                target_density=source["target_density"],
                rate=source["rate"],
            )

        current_readings = sim.get_sensor_data()
        sensor_readings_history.append(current_readings)

        vel_mag = np.linalg.norm(current_readings, axis=1)
        if np.isnan(vel_mag).any() or np.max(vel_mag) > max_expected_velocity:
            print(
                f"Aborting run {run_id:04d} at step {t} due to invalid sensor readings."
            )
            shutil.rmtree(run_dir)
            return None

        sensor_sum += current_readings.sum(axis=0)
        sensor_sum_sq += (current_readings**2).sum(axis=0)
        sensor_count += current_readings.shape[0]

        if t % snapshot_interval == 0:
            smoke_img, vel = sim.get_data_snapshot()

            smoke_sum += smoke_img.sum()
            smoke_sum_sq += (smoke_img**2).sum()
            smoke_count += smoke_img.size

            np.save(
                os.path.join(raw_data_dir, f"smoke_{t:05d}.npy"),
                smoke_img.astype(np.float16),
            )
            if generate_video:
                smoke_t = smoke_img.T
                mask_t = mask.T

                norm_smoke = np.clip(smoke_t * 255, 0, 255).astype(np.uint8)
                norm_smoke[mask_t == 1] = 80
                frames_smoke.append(
                    Image.fromarray(norm_smoke, mode="L").convert("RGB")
                )

                vel_frame = velocity_field_to_image(
                    vel, sensor_positions, current_readings, mask, sim.nx, sim.ny
                )
                frames_velocity.append(vel_frame)

    if generate_video:
        save_gif(
            frames_smoke,
            os.path.join(run_dir, f"smoke_simulation_{run_id:04d}.gif"),
            fps=15,
        )
        save_gif(
            frames_velocity,
            os.path.join(run_dir, f"velocity_simulation_{run_id:04d}.gif"),
            fps=15,
        )

    saved_config = config.copy()
    saved_config["run_id"] = run_id

    np.savez_compressed(
        os.path.join(run_dir, "metadata.npz"),
        input_profile=wind_profile,
        sensor_readings=np.array(sensor_readings_history),
        sensor_positions=np.array(sensor_positions),
        mask=mask,
        sources=np.array(smoke_sources),
        env_config=np.array(saved_config),
    )

    return {
        "sensor_sum": sensor_sum,
        "sensor_sum_sq": sensor_sum_sq,
        "sensor_count": sensor_count,
        "smoke_sum": smoke_sum,
        "smoke_sum_sq": smoke_sum_sq,
        "smoke_count": smoke_count,
    }


def main(args):
    config_file = args.config_file
    with open(config_file, "r") as f:
        config = json.load(f)

    width = config["width"]
    height = config["height"]
    viscosity = config["viscosity"]
    diffusivity = config["diffusivity"]
    num_sensors = config["num_sensors"]
    total_steps = config["total_steps"]
    snapshot_interval = config["snapshot_interval"]
    seed = config["seed"]
    num_runs = config["num_runs"]
    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]
    test_ratio = config["test_ratio"]
    inlet_profile_type = config["inlet_profile_type"]
    output_root = config["output_root"]

    ti.init(arch=ti.cuda)

    train_end = int(train_ratio * num_runs)
    val_end = train_end + int(val_ratio * num_runs)

    sim_config = {
        "width": width,
        "height": height,
        "viscosity": viscosity,
        "diffusivity": diffusivity,
        "num_sensors": num_sensors,
        "total_steps": total_steps,
        "snapshot_interval": snapshot_interval,
        "seed": seed,
        "inlet_profile_type": inlet_profile_type,
    }

    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "test"), exist_ok=True)

    pbar = trange(num_runs, desc="Overall Simulation Runs")
    metadata = {
        "sim_config": sim_config,
        "num_runs": num_runs,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_indices": {
            "train": [],
            "val": [],
            "test": [],
        },
    }

    sensor_sum_total = np.zeros(2, dtype=np.float64)
    sensor_sum_sq_total = np.zeros(2, dtype=np.float64)
    sensor_count_total = 0

    smoke_sum_total = 0.0
    smoke_sum_sq_total = 0.0
    smoke_count_total = 0

    successful_runs = 0
    run_id = 0

    while successful_runs < num_runs:
        if successful_runs < train_end:
            subset = "train"
            generate_video = False
        elif successful_runs < val_end:
            subset = "val"
            generate_video = False
        else:
            subset = "test"
            generate_video = True

        target_dir = os.path.join(output_root, subset)

        run_stats = run_single_simulation(
            run_id, sim_config, target_dir, generate_video
        )

        if run_stats is not None:
            metadata["split_indices"][subset].append(run_id)

            sensor_sum_total += run_stats["sensor_sum"]
            sensor_sum_sq_total += run_stats["sensor_sum_sq"]
            sensor_count_total += run_stats["sensor_count"]

            smoke_sum_total += run_stats["smoke_sum"]
            smoke_sum_sq_total += run_stats["smoke_sum_sq"]
            smoke_count_total += run_stats["smoke_count"]

            successful_runs += 1
            pbar.update(1)

        run_id += 1

    pbar.close()

    sensor_mean = (sensor_sum_total / sensor_count_total).tolist()
    sensor_var = (sensor_sum_sq_total / sensor_count_total) - (
        sensor_sum_total / sensor_count_total
    ) ** 2
    sensor_std = np.sqrt(np.maximum(sensor_var, 1e-6)).tolist()

    smoke_mean = smoke_sum_total / smoke_count_total
    smoke_var = (smoke_sum_sq_total / smoke_count_total) - (smoke_mean**2)
    smoke_std = np.sqrt(max(smoke_var, 1e-6))

    smoke_mean = float(smoke_mean)
    smoke_std = float(smoke_std)

    metadata["sensor_stats"] = {"mean": sensor_mean, "std": sensor_std}
    metadata["smoke_stats"] = {"mean": smoke_mean, "std": smoke_std}

    print("---- DATASET GENERATION COMPLETE ----")
    print("Sensor Mean:", sensor_mean)
    print("Sensor Std:", sensor_std)
    print("Smoke Mean:", smoke_mean)
    print("Smoke Std:", smoke_std)

    with open(os.path.join(output_root, "dataset_summary.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def modify_parser(parser):
    parser.add_argument(
        "--config_file",
        type=str,
        default="dataset_config.json",
        help="Path to configuration file",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modify_parser(parser)
    args = parser.parse_args()
    main(args)
