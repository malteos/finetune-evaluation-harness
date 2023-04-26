from unittest.mock import MagicMock
from hf_scripts.utility_functions import *
import os
import pytest
import unittest
from process_args import process_arguments
from transformers import HfArgumentParser, TrainingArguments
from hf_scripts.data_trainining_args import DataTrainingArguments
from tasks import *
import tasks
from hf_scripts.hgf_fine_tune_class import *
from hf_scripts.hgf_fine_tune_ner import *
from hf_scripts.hgf_fine_tune_qa import *


"""
File checkin the consistency of the recent push made to the repo (previous code doesnt break)
"""

def test_freeze():
    model_args = ModelArguments(model_name_or_path="bert-base-german-cased")
    model_args.freeze_layers = True
    model = AutoModelForTokenClassification.from_pretrained("bert-base-german-cased")
    assert freeze_layers(model_args, model) == model



def test_add_labels_args():
    data_args = DataTrainingArguments()
    sample_task = "germeval2018"
    TASK_REGISTRY = {"germeval2018": germeval2018.GermEval2018}
    germeval_obj = germeval2018.GermEval2018()
    germeval_obj.get_dataset_id = MagicMock(return_value="philschmid/germeval18")
    germeval_obj.get_task_type = MagicMock(return_value="classification")
    germeval_obj.get_label_name = MagicMock(return_value="label")
    assert add_labels_data_args(sample_task, data_args) == data_args

    

def test_tasks_initialization():
    test_task = "german_ner_legal"
    assert isinstance(get_all_tasks(), list)

    