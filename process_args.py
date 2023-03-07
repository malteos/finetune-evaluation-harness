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

def call_hf_script(args, task_name, peft_choice):

    """
    method to call hf_script method based on type and task_name
    """
    task_type = TASK_TYPE_REGISTRY[task_name]
    task_class_obj = TASK_REGISTRY[task_name]
    converted_dict = vars(args).copy()
    script_arguments = []

    # change the parameter name to one compatible with hf_script by adding a few extra parameters
    converted_dict['model_name_or_path'] = converted_dict.pop('model')
    converted_dict['dataset_name'] = converted_dict.pop('task_list')
    converted_dict['dataset_name'] = converted_dict['dataset_name'][0]
    converted_dict['output_dir'] = converted_dict.pop('base_checkpoint_dir')
    converted_dict['num_train_epochs'] = converted_dict.pop('epochs')
    converted_dict['peft_choice'] = peft_choice
    converted_dict['results_log_path']  = converted_dict.pop('results_logging_dir')
    converted_dict['dataset_name'] = task_class_obj().get_dataset_id()
    

    for key, value in converted_dict.items():
        updated_key = '--'+ str(key)+""
        script_arguments.append(updated_key)
        script_arguments.append(value)
    
    script_arguments.append('--do_train')
    script_arguments.append('--do_eval')
    if(task_type == 'ner'):
        script_arguments.append("--is_task_ner")
        script_arguments.append("True")

    map_source_file(task_name).main(script_arguments)
    #map_source_file(task_name).handle_arguments(script_arguments)
    del converted_dict



def process_arguments(args):
    """
    main method accepts argaprse arguments and process it to utilize the training scripts
    """

    if(args.task_list == ["ALL"]):
        tasks_to_run = get_all_tasks()
    else:
        tasks_to_run = args.task_list

    if(args.peft_choice == ['ALL']):
        peft_to_run = ['lora', 'p_tune', 'prefix_tune', 'prompt_tune']
    else:
        peft_to_run = args.peft_choice


    if(len(tasks_to_run)==1 and peft_to_run is not None):
        for each_peft in peft_to_run:        
            call_hf_script(args, args.task_list[0], each_peft)
    else:
        for each_task in tasks_to_run:
            call_hf_script(args, each_task, args.peft_choice)



