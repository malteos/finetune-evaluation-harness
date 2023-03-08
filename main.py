import datasets
from math import exp
import sys
#from hf_scripts import hgf_fine_tune_class, hgf_fine_tune_ner, hgf_fine_tune_qa
from tasks import get_all_tasks, TASK_REGISTRY, TASK_TYPE_REGISTRY, get_all_task_types
import argparse
from hf_scripts.model_args import ModelArguments
from process_args import process_arguments
from transformers import (HfArgumentParser, TrainingArguments)
from hf_scripts.model_args import ModelArguments
from hf_scripts.data_trainining_args import DataTrainingArguments
from hf_scripts.initial_arguments import InitialArguments

## main script for runing the tasks
## Indivivual tasks defined in the /tasks folder


def main():
    process_arguments(sys.argv[1:])

if __name__ == "__main__":
    main()
