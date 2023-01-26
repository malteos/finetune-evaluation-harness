import datasets
from math import exp
from hf_scripts import hgf_fine_tune_class, hgf_fine_tune_ner, hgf_fine_tune_qa
from tasks import get_all_tasks, TASK_REGISTRY, TASK_TYPE_REGISTRY, get_all_task_types
import argparse
import logging
import json 
import fnmatch

## main script for runing the tasks
## Indivivual tasks defined in the /tasks folder

BASE_CHECKPOINT_DIR = "/netscratch/agautam/experiments/test_logs"

# name of the file used for recording metrics in json
RESULTS_LOG_DIR = "/netscratch/agautam/experiments/test_logs/metrics"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task_list", nargs='+', default=None)
    parser.add_argument("--freeze_layers", default=True)
    parser.add_argument("--base_checkpoint_dir", default = BASE_CHECKPOINT_DIR)
    parser.add_argument("--results_logging_dir", default = RESULTS_LOG_DIR)
    parser.add_argument("--epochs", default = "1")
    parser.add_argument("--per_device_train_batch_size", default = "2")
    parser.add_argument("--save_steps", default = "15000")

    return parser.parse_args()


# identify the classiification type and return the file huggingface source script name
def map_source_file(task_name):
    task_type = TASK_TYPE_REGISTRY[task_name]
    if(task_type == "classification"):
        return hgf_fine_tune_class
    if(task_type == "ner"):
        return hgf_fine_tune_ner
    else:
        return hgf_fine_tune_qa


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

def main():
    args = parse_args()
    print(args.task_list)
    #task_names = pattern_match(args.tasks.split(","), ALL_TASK_TYPES)
    #print(f"Selected Tasks: {task_names}")

    if(args.task_list == ["ALL"]):
        tasks_to_run = get_all_tasks()
    else:
        tasks_to_run = args.task_list   

    print("tasks_to_run", tasks_to_run)
    for each_task in tasks_to_run:
        task_class_obj = TASK_REGISTRY[each_task]
        print(task_class_obj)
        all_params = task_class_obj().create_train_params(
            args.model, 
            args.base_checkpoint_dir, 
            args.results_logging_dir, 
            args.freeze_layers, 
            args.epochs,
            args.per_device_train_batch_size,
            args.save_steps
        )
        map_source_file(each_task).main(all_params)

if __name__ == "__main__":
    main()