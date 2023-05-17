from unittest.mock import MagicMock
import os
import pytest
import unittest
#from src.finetune_eval.process_args import process_arguments
#import src.finetune_eval.process_args
from transformers import HfArgumentParser, TrainingArguments,AutoModelForTokenClassification

# from src.finetune_eval.hf_scripts.data_trainining_args import DataTrainingArguments
# from src.finetune_eval.tasks import *
# import src.finetune_eval.tasks
# from src.finetune_eval.hf_scripts.hgf_fine_tune_class import *
# from src.finetune_eval.hf_scripts.hgf_fine_tune_ner import *
# from src.finetune_eval.hf_scripts.hgf_fine_tune_qa import *
# from src.finetune_eval.hf_scripts.model_args import ModelArguments
# from src.finetune_eval.hf_scripts.data_trainining_args import DataTrainingArguments
# from src.finetune_eval.hf_scripts.initial_arguments import InitialArguments
# from src.finetune_eval.hf_scripts import hgf_fine_tune_class, hgf_fine_tune_ner, hgf_fine_tune_qa

from finetune_eval_harness.hf_scripts.data_trainining_args import DataTrainingArguments
from finetune_eval_harness.hf_scripts import hgf_fine_tune_class, hgf_fine_tune_ner, hgf_fine_tune_qa
from finetune_eval_harness.hf_scripts.model_args import ModelArguments
from finetune_eval_harness.hf_scripts.initial_arguments import InitialArguments
from finetune_eval_harness.hf_scripts.utility_functions import freeze_layers, add_labels_data_args, get_all_tasks
from finetune_eval_harness.tasks import germeval2018


"""
File checkin the consistency of the recent push made to the repo (previous code doesnt break)
"""

def test_freeze():
    model_args = ModelArguments(model_name_or_path="bert-base-german-cased")
    model_args.freeze_layers = True
    model = AutoModelForTokenClassification.from_pretrained("bert-base-german-cased")
    assert freeze_layers(model_args, model) == model
    #assert isinstance(model_args, ModelArguments)



def test_add_labels_args():
    data_args = DataTrainingArguments()
    sample_task = "germeval2018"
    TASK_REGISTRY = {"germeval2018": germeval2018.GermEval2018}
    germeval_obj = germeval2018.GermEval2018()
    germeval_obj.get_dataset_id = MagicMock(return_value="philschmid/germeval18")
    germeval_obj.get_task_type = MagicMock(return_value="classification")
    germeval_obj.get_label_name = MagicMock(return_value="label")
    assert add_labels_data_args(sample_task, data_args) == data_args
    #assert isinstance(sample_task, str)

    

def test_tasks_initialization():
    test_task = "german_ner_legal"
    assert isinstance(get_all_tasks(), list)

    