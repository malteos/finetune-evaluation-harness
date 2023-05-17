

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

'''
import sys 
sys.path.append('../')
from src.finetune_eval_harness.tasks.task_registry import get_all_tasks, get_all_task_types, get_dataset_information

import src.finetune_eval_harness.tasks
import src.finetune_eval_harness.hf_scripts
import src.finetune_eval_harness.tasks as tasks
import src.finetune_eval_harness.hf_scripts as hf_scripts
'''

import sys
sys.path.append('./')
from .tasks import *
from .tasks.task_registry import get_all_tasks, TASK_REGISTRY, TASK_TYPE_REGISTRY, get_all_task_types, get_dataset_information
from .hf_scripts import *

sys.path.append('../')
#import hf_scripts
