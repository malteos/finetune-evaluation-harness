from transformers import HfArgumentParser, TrainingArguments
from tasks import get_all_tasks, TASK_REGISTRY, TASK_TYPE_REGISTRY
from hf_scripts.utility_functions import (
    map_source_file,
    peft_choice_list,
    add_labels_data_args,
)
from hf_scripts.model_args import ModelArguments
from hf_scripts.data_trainining_args import DataTrainingArguments
from hf_scripts.initial_arguments import InitialArguments
import logging


def process_arguments(args):
    """
    main method accepts argaprse arguments and process it to utilize the training scripts
    """

    logging.info(f"args {args}")
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    logging.info(f"parser {parser}")
    parser._add_dataclass_arguments(InitialArguments)
    (
        model_args,
        data_args,
        training_args,
        init_args,
    ) = parser.parse_args_into_dataclasses(args=args)

    task_list = init_args.task_list
    logging.info(f"task_list {task_list}")

    if task_list == ["ALL"]:
        tasks_to_run = get_all_tasks()
    else:
        tasks_to_run = task_list
    

    if len(tasks_to_run) == 1 and data_args.peft_choice in peft_choice_list:
        data_args = add_labels_data_args(tasks_to_run[0], data_args)
        map_source_file(tasks_to_run[0]).run_task_evaluation(
            model_args, data_args, training_args, init_args
        )

    else:
        for each_task in tasks_to_run:
            data_args = add_labels_data_args(each_task, data_args)
            map_source_file(each_task).run_task_evaluation(
                model_args, data_args, training_args, init_args
            )

    return parser
