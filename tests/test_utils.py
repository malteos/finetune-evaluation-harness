# Imports in this file are based on module usages after pip install

from unittest.mock import MagicMock
import os
import pytest
import unittest
import tempfile

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoModelForTokenClassification,
    AutoConfig,
    AutoModelForQuestionAnswering,
)

from finetune_eval_harness.hf_scripts.data_trainining_args import DataTrainingArguments
from finetune_eval_harness.hf_scripts import (
    hgf_fine_tune_class,
    hgf_fine_tune_ner,
    hgf_fine_tune_qa,
)
from finetune_eval_harness.hf_scripts.model_args import ModelArguments
from finetune_eval_harness.hf_scripts.initial_arguments import InitialArguments
from finetune_eval_harness.hf_scripts.utility_functions import (
    freeze_layers,
    add_labels_data_args,
    get_all_tasks,
    prepend_data_args,
    load_model,
    load_config,
    map_source_file,
    get_label_list,
)
from finetune_eval_harness.tasks import *


"""
File consisting of integeration unit test cases for utility functions (implemented logic)
"""


def test_add_labels_args():
    data_args = DataTrainingArguments()
    sample_task = "germeval2018"
    TASK_REGISTRY = {"germeval2018": germeval2018.GermEval2018}
    germeval_obj = germeval2018.GermEval2018()
    germeval_obj.get_dataset_id = MagicMock(return_value="philschmid/germeval18")
    germeval_obj.get_task_type = MagicMock(return_value="classification")
    germeval_obj.get_label_name = MagicMock(return_value="label")
    assert add_labels_data_args(sample_task, data_args) == data_args


def test_prepend_data_args():
    data_args = DataTrainingArguments()
    temp_dir_name = tempfile.TemporaryDirectory().name
    training_args = TrainingArguments(output_dir=temp_dir_name)
    init_args = InitialArguments()
    # init_args.results_logging_dir = "/sample/directory"
    init_args.results_logging_dir = temp_dir_name
    # training_args.output_dir = "/sample/directory"
    training_args.output_dir = temp_dir_name
    assert prepend_data_args(training_args, data_args, init_args) == (
        training_args,
        data_args,
    )


@pytest.mark.skip()
def test_freeze():
    model_path = os.getcwd() + "/tests/custom_model"
    model_args = ModelArguments(model_name_or_path=model_path)
    model_args.freeze_layers = True
    model = AutoModelForTokenClassification.from_pretrained("bert-base-german-cased")
    assert freeze_layers(model_args, model) == model

@pytest.mark.skip()
def test_load_config():
    temp_dir_name = tempfile.TemporaryDirectory().name
    model_path = os.getcwd() + "/tests/custom_model"
    # model_name_or_path = "bert-base-german-cased"
    model_name_or_path = model_path
    num_labels = 1
    finetuning_task = "classification"
    cache_dir = temp_dir_name
    # cache_dir = "/sample/directory"
    model_revision = "main"
    use_auth_token = False
    model_type = "classification"
    data_args = DataTrainingArguments()

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
            data_args,
        )
        == config
    )


def test_load_model():
    model_path = os.getcwd() + "/tests/custom_model"
    temp_dir_name = tempfile.TemporaryDirectory().name
    # model_name_or_path = "bert-base-german-cased"
    model_name_or_path = model_path
    finetuning_task = "question-answering"
    cache_dir = temp_dir_name
    # cache_dir = "/sample/directory"
    model_revision = "main"
    use_auth_token = False
    model_type = "qa"
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        fine_tuning_task=finetuning_task,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=use_auth_token,
    )
    use_fast = False
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path,
        from_tf=False,
        # config=config,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=use_auth_token,
    )

    assert (
        load_model(
            model_name_or_path,
            False,
            config,
            cache_dir,
            model_revision,
            False,
            "qa",
        ).base_model_prefix
        == "bert"
    )


def test_tasks_initialization():
    test_task = "german_ner_legal"
    assert isinstance(get_all_tasks(), list)


def test_get_labels():
    label_list = ["binary", "multi"]
    assert isinstance(get_label_list(label_list), list)


def test_map_source_file():
    task_name = "gnad10"
    assert type(map_source_file(task_name))


