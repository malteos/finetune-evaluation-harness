import datasets
from math import exp
#from hf_scripts import hgf_fine_tune_class, hgf_fine_tune_ner, hgf_fine_tune_qa
from tasks import get_all_tasks, TASK_REGISTRY, TASK_TYPE_REGISTRY, get_all_task_types
import argparse
from hf_scripts.model_args import ModelArguments
from process_args import process_arguments

## main script for runing the tasks
## Indivivual tasks defined in the /tasks folder


def parse_args():

    """
    Argparse for regular usage of the evaluation platform
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="name of the model")
    parser.add_argument(
        "--task_list", nargs="+", default=None, help="list of the tasks"
    )
    parser.add_argument(
        "--freeze_layers",
        default=False,
        help="do you want to freeze the layers of the models",
    )
    parser.add_argument(
        "--base_checkpoint_dir",
        required=True,
        help="path where the model checkpoints would be saved",
    )
    parser.add_argument(
        "--results_logging_dir",
        required=True,
        help="path where the results json file will be saved",
    )
    parser.add_argument("--epochs", default="1", help="number of epochs to run")
    parser.add_argument(
        "--per_device_train_batch_size", default="8", help="the size of train batch"
    )
    parser.add_argument(
        "--save_steps",
        default="15000",
        help="number of steps after which to save the model checkpoints",
    )
    parser.add_argument(
        "--peft_choice",
        nargs="+",
        default=None,
        help="choice of parameter efficent fine tuning",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        default="True",
        help="overwrite the directory where model weights will be saved",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    process_arguments(args)


if __name__ == "__main__":
    main()
