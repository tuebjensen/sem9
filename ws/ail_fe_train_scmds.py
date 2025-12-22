import json

from ail_fe_main_scmds import SCmd


def modify_parser(parser):
    parser.add_argument(
        "--experiments_path",
        type=str,
        default="experiments.json",
        help="Path to experiments configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="parallel",
        help="Mode to run experiments: sequential, parallel",
    )


def get_scmd(experiment):
    run_name = experiment["run_name"]
    image_head_type = experiment["image_head_type"]
    upsampling_type = experiment["upsampling_type"]
    image_loss_type = experiment["image_loss_type"]
    return SCmd(
        opts=[
            "-J",
            run_name,
            "--gres=gpu:1",
            "--cpus-per-task=10",
            "--mem=50G",
            "--time=12:00:00",
            f"--output=logs/{run_name}_%j.out",
            f"--error=logs/{run_name}_%j.err",
        ],
        python_module="train",
        python_args=[
            "--run_name",
            run_name,
            "--image_head_type",
            image_head_type,
            "--upsampling_type",
            upsampling_type,
            "--image_loss_type",
            image_loss_type,
        ],
    )


def get_scmds(args):
    with open(args.experiments_path, "r") as f:
        experiments_dict = json.load(f)

    scmds = []
    for experiment in experiments_dict["experiments"]:  
        scmd = get_scmd(experiment)
        scmds.append(scmd)

    return scmds
