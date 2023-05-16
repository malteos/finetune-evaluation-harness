

## please make sure that the vesion numbers are same here and setup.py

#__version__ = "0.6.0.dev0"

'''
from .hf_scripts import (
    data_trainining_args,
    hgf_fine_tune_class,
    hgf_fine_tune_ner,
    hgf_fine_tune_qa,
    initial_arguments,
    model_args,
    trainer_qa,
    utility_functions,
    utils_qa,
)

from .tasks import (
    classification,
    german_europarl,
    german_ner,
    germeval2017,
    germeval2018,
    gnad10,
    ner,
    qa,
    german_quad,
)
'''
import sys 
sys.path.append('../')
#from . import process_args
#from . import main
from src.finetune_eval_harness.tasks.task_registry import get_all_tasks, get_all_task_types, get_dataset_information
#from .process_args import process_arguments
#import src.finetune_eval.process_args as process_args


#import src.finetune_eval.main as main
import src.finetune_eval_harness.tasks
import src.finetune_eval_harness.hf_scripts

