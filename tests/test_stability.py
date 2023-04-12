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


#@pytest.mark.skip()
def test_load_config():
    model_name_or_path = "bert-base-german-cased"
    num_labels = 1
    finetuning_task = "classification"
    cache_dir = "/sample/directory"
    model_revision = "main"
    use_auth_token = False
    model_type = "classification"

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        fine_tuning_task=finetuning_task,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=use_auth_token,
    )
    assert (
        load_config(
            model_name_or_path,
            num_labels,
            finetuning_task,
            cache_dir,
            model_revision,
            use_auth_token,
            model_type,
        )
        == config
    )

def test_add_labels_args():
    data_args = DataTrainingArguments()
    sample_task = "germeval2018"
    TASK_REGISTRY = {"germeval2018": germeval2018.GermEval2018}
    germeval_obj = germeval2018.GermEval2018()
    germeval_obj.get_dataset_id = MagicMock(return_value="philschmid/germeval18")
    germeval_obj.get_task_type = MagicMock(return_value="classification")
    germeval_obj.get_label_name = MagicMock(return_value="label")
    assert add_labels_data_args(sample_task, data_args) == data_args