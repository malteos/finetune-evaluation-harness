import sys

import json
import sys

from transformers import HfArgumentParser, TrainingArguments, set_seed

from tasks.task_registry import get_all_tasks, TASK_REGISTRY
import logging

# from hf_scripts.utility_functions import (
#     peft_choice_list,
# )

from hf_scripts.model_args import ModelArguments
from hf_scripts.data_trainining_args import DataTrainingArguments
from hf_scripts.initial_arguments import InitialArguments

from hf_scripts import utility_functions

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_arguments(args):
    """
    main method accepts argaprse arguments and process it to utilize the training scripts
    """

    logger.info(f"args {args}")
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    logger.info(f"parser {parser}")
    parser._add_dataclass_arguments(InitialArguments)
    (
        model_args,
        data_args,
        training_args,
        init_args,
    ) = parser.parse_args_into_dataclasses(args=args)

    if init_args.tasks == "ALL":
        tasks_to_run = get_all_tasks()
    else:
        tasks_to_run = init_args.tasks.split(",")

    logger.info(f"Tasks to evaluate: {tasks_to_run}")

    logger.info(f"Training Args: {training_args}")

    for task_i, task_name in enumerate(tasks_to_run, 1):
        logger.info(f"Current task: {task_name} ({task_i}/{len(tasks_to_run)})")

        task_cls = TASK_REGISTRY[task_name]

        task = task_cls(model_args, data_args, training_args, init_args)
        task_metrics = task.evaluate()

        logger.info(
            f"Task metrics {json.dumps(task_metrics, sort_keys=True, indent=4)}"
        )

    return parser


if __name__ == "__main__":
    process_arguments(sys.argv[1:])
