from hf_scripts import *
import argparse
import logging
import json
import os
import sys  
import fnmatch
from hf_scripts.model_args import ModelArguments
from tasks import get_all_tasks, TASK_REGISTRY, TASK_TYPE_REGISTRY, get_all_task_types
from hf_scripts import hgf_fine_tune_class, hgf_fine_tune_ner, hgf_fine_tune_qa
from hf_scripts.utility_functions import map_source_file
from hf_scripts.utility_functions import parse_hf_arguments
from transformers import (HfArgumentParser, TrainingArguments)
from hf_scripts.model_args import ModelArguments
from hf_scripts.data_trainining_args import DataTrainingArguments
from hf_scripts.initial_arguments import InitialArguments
from hf_scripts.utility_functions import peft_choice_list


def process_arguments(args):
    """
    main method accepts argaprse arguments and process it to utilize the training scripts
    """

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser._add_dataclass_arguments(InitialArguments)
    model_args, data_args, training_args, init_args = parser.parse_args_into_dataclasses(args = args)
    
    task_list = init_args.task_list
    print(task_list)
    if(task_list == ["ALL"]):
        tasks_to_run = get_all_tasks()
    else:
        tasks_to_run = task_list
    
    if(len(tasks_to_run)== 1 and data_args.peft_choice in peft_choice_list):
        dataset_name = TASK_REGISTRY[tasks_to_run[0]]().get_dataset_id()
        label_value = TASK_REGISTRY[tasks_to_run[0]]().get_label_name()
        data_args.dataset_name = dataset_name
        data_args.label_value = label_value
        map_source_file(tasks_to_run[0]).run_task_evaluation(model_args, data_args, training_args, init_args)
    else:
        for each_task in tasks_to_run:
            dataset_name = TASK_REGISTRY[each_task]().get_dataset_id()
            label_value = TASK_REGISTRY[each_task]().get_label_name()
            data_args.dataset_name = dataset_name
            data_args.label_value = label_value
            map_source_file(each_task).run_task_evaluation(model_args, data_args, training_args, init_args)
    



