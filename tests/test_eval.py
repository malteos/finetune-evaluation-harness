from unittest.mock import MagicMock
from hf_scripts.utility_functions import *
import os
import pytest
import unittest
import process_args
from transformers import HfArgumentParser, TrainingArguments
from hf_scripts.data_trainining_args import DataTrainingArguments
from tasks import *
from hf_scripts.hgf_fine_tune_class import *
from hf_scripts.hgf_fine_tune_ner import *
from hf_scripts.hgf_fine_tune_qa import *
import tempfile

"""
File consisting of integeration unit test cases for evaluating each of the tasks
"""


#@pytest.mark.skip()
def test_cls_evaluation():
    temp_dir_name = tempfile.TemporaryDirectory().name
    data_args = DataTrainingArguments()
    data_args.dataset_name = "akash418/germeval_2017"
    #data_args.base_checkpoint_dir = "/tmp/directory"
    data_args.base_checkpoint_dir = temp_dir_name
    #print("temp directory ....", tempfile.TemporaryDirectory().name)
    data_args.is_task_ner = False
    data_args.label_value = "relevance"
    data_args.peft_choice = "p_tune"

    model_args = ModelArguments("bert-base-german-cased")
    model_args.model_name_or_path = "bert-base-german-cased"
    model_args.model_revision = "main"
    model_args.use_fast_tokenizer = False
    
    training_args = TrainingArguments(output_dir = temp_dir_name)
    #training_args = TrainingArguments(output_dir="/tmp/directory")
    #training_args.output_dir = "/tmp/directory"
    training_args.output_dir = temp_dir_name
    training_args.num_train_epochs = 0
    training_args.do_train = True
    training_args.do_eval = True
    training_args.do_predict = True
    training_args.per_device_train_batch_size = 128

    init_args = InitialArguments()
    #init_args.results_logging_dir = "/tmp/directory"
    init_args.results_logging_dir = temp_dir_name
    init_args.task_list = "germeval2017"

    metrics_eval = hgf_fine_tune_class.run_task_evaluation(model_args, data_args, training_args, init_args)
    assert metrics_eval['eval_accuracy'] == pytest.approx(0.81, 0.3)


@pytest.mark.skip()
def test_ner_evaluation():
    data_args = DataTrainingArguments()
    data_args.dataset_name = "elenanereiss/german-ler"
    data_args.base_checkpoint_dir = "/tmp/directory"
    data_args.is_task_ner = True
    # data_args.label_value="multi"
    data_args.peft_choice = "prefix_tune"
    data_args.return_entity_level_metrics = True

    model_args = ModelArguments("bert-base-german-cased")
    model_args.model_name_or_path = "bert-base-german-cased"
    model_args.model_revision = "main"
    model_args.use_fast_tokenizer = True

    training_args = TrainingArguments(output_dir="/tmp/directory")
    training_args.output_dir = "/tmp/directory"
    training_args.num_train_epochs = 1
    training_args.do_train = True
    training_args.do_eval = True
    #training_args.do_predict = True
    training_args.per_device_train_batch_size=64

    init_args = InitialArguments()
    init_args.results_logging_dir = "/tmp/directory"
    init_args.task_list = "german_ner"

    metrics_eval = hgf_fine_tune_ner.run_task_evaluation(model_args, data_args, training_args, init_args)
    assert metrics_eval['eval_overall_accuracy'] == pytest.approx(0.90, 0.3)


@pytest.mark.skip()
def test_qa_evaluation():
    data_args = DataTrainingArguments()
    data_args.dataset_name = "deepset/germanquad"
    data_args.base_checkpoint_dir = "/tmp/directory"
    data_args.is_task_ner = False
    # data_args.label_value="multi"
    data_args.peft_choice = "prompt_tune"
    data_args.version_2_with_negative = True

    model_args = ModelArguments("bert-base-german-cased")
    model_args.model_name_or_path = "bert-base-german-cased"
    model_args.model_revision = "main"
    model_args.use_fast_tokenizer = True

    training_args = TrainingArguments(output_dir="/tmp/directory")
    training_args.output_dir = "/tmp/directory"
    training_args.num_train_epochs = 1
    training_args.do_train = True
    training_args.do_eval = True
    training_args.do_predict = True
    training_args.overwrite_output_dir = True

    init_args = InitialArguments()
    init_args.results_logging_dir = "/tmp/directory"
    init_args.task_list = "german_quad"


    metrics_eval = hgf_fine_tune_qa.run_task_evaluation(model_args, data_args, training_args, init_args)
    assert metrics_eval['eval_f1'] == pytest.approx(50.99, 0.3)

