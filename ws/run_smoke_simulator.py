import argparse
import os
import subprocess
import sys


def main(args):
    local_lib_path = "/ceph/home/student.aau.dk/pw66uf/tmp/usr/lib/x86_64-linux-gnu/"

    env = os.environ.copy()

    # Workaround for missing library files on AI-LAB
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{local_lib_path}:{current_ld}"

    cmd = [sys.executable, "smoke_simulator.py"] + sys.argv[1:]

    print(f"--- Launching Simulation ---")
    print(f"Library Path: {local_lib_path}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 30)

    try:
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nProcess interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching script: {e}")
        sys.exit(1)


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
