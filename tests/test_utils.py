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

#@pytest.mark.skip()
def test_add_labels_args():
    data_args = DataTrainingArguments()
    sample_task = "germeval2018"
    TASK_REGISTRY = {"germeval2018": germeval2018.GermEval2018}
    germeval_obj = germeval2018.GermEval2018()
    germeval_obj.get_dataset_id = MagicMock(return_value = "philschmid/germeval18")
    germeval_obj.get_task_type = MagicMock(return_value = "classification")
    germeval_obj.get_label_name = MagicMock(return_value = "label")
    assert add_labels_data_args(sample_task, data_args) == data_args

#@pytest.mark.skip()
def test_prepend_data_args():
    data_args = DataTrainingArguments()
    training_args = TrainingArguments(output_dir = "/sample/directory")
    init_args = InitialArguments()
    init_args.results_logging_dir = "/sample/directory"
    training_args.output_dir = "/sample/directory"
    assert prepend_data_args(training_args, data_args, init_args) == (training_args, data_args)

#@pytest.mark.skip()
def test_freeze():
    model_args = ModelArguments(model_name_or_path='bert-base-german-cased')
    model_args.freeze_layers = True
    model = AutoModelForTokenClassification.from_pretrained('bert-base-german-cased')
    assert freeze_layers(model_args, model) == model 

#@pytest.mark.skip()
def test_load_config():
    model_name_or_path = "bert-base-german-cased"
    num_labels = 1
    finetuning_task ="classification"
    cache_dir = "/sample/directory"
    model_revision = "main"
    use_auth_token = False
    model_type = "classification"

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels = num_labels,
        fine_tuning_task = finetuning_task,
        cache_dir = cache_dir,
        revision = model_revision,
        use_auth_token = use_auth_token,
    )
    assert load_config(
        model_name_or_path, 
        num_labels, 
        finetuning_task, 
        cache_dir,
        model_revision,
        use_auth_token,
        model_type
    ) == config

#@pytest.mark.skip()
def test_load_model():
    model_name_or_path = "bert-base-german-cased"
    finetuning_task ="question-answering"
    cache_dir = "/sample/directory"
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
    model =  AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            from_tf=False,
            #config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=use_auth_token,
        )
    
    
    assert load_model(
        model_name_or_path,
        False,
        config,
        cache_dir,
        model_revision,
        False,
        "qa",
    ).base_model_prefix == "bert"

#@pytest.mark.skip()
def test_cls_evaluation():
    data_args = DataTrainingArguments()
    data_args.dataset_name = "philschmid/germeval18"
    data_args.base_checkpoint_dir="/sample/directory"
    data_args.is_task_ner=False
    data_args.label_value="multi"
    data_args.peft_choice=None
    
    model_args = ModelArguments("bert-base-german-cased")
    model_args.model_name_or_path = "bert-base-german-cased"
    model_args.model_revision="main"
    model_args.use_fast_tokenizer=True

    training_args = TrainingArguments(output_dir = "/sample/directory")
    training_args.output_dir = "/sample/directory"
    training_args.num_train_epochs = 1

    init_args = InitialArguments()
    init_args.results_logging_dir="/sample/directory"
    init_args.task_list="germeval2018"

    assert isinstance(hgf_fine_tune_class.run_task_evaluation(model_args, data_args, training_args, init_args), Trainer)

#@pytest.mark.skip()
def test_ner_evaluation():
    data_args = DataTrainingArguments()
    data_args.dataset_name = "akash418/german_europarl"
    data_args.base_checkpoint_dir="/sample/directory"
    data_args.is_task_ner=True
    #data_args.label_value="multi"
    data_args.peft_choice=None
    
    model_args = ModelArguments("bert-base-german-cased")
    model_args.model_name_or_path = "bert-base-german-cased"
    model_args.model_revision="main"
    model_args.use_fast_tokenizer=True

    training_args = TrainingArguments(output_dir = "/sample/directory")
    training_args.output_dir = "/sample/directory"
    training_args.num_train_epochs = 1

    init_args = InitialArguments()
    init_args.results_logging_dir="/sample/directory"
    init_args.task_list="german_europarl"


    assert isinstance(hgf_fine_tune_ner.run_task_evaluation(model_args, data_args, training_args, init_args), Trainer)


def test_qa_evaluation():
    data_args = DataTrainingArguments()
    data_args.dataset_name = "deepset/germanquad"
    data_args.base_checkpoint_dir="/sample/directory"
    data_args.is_task_ner=False
    #data_args.label_value="multi"
    data_args.peft_choice=None
    
    model_args = ModelArguments("bert-base-german-cased")
    model_args.model_name_or_path = "bert-base-german-cased"
    model_args.model_revision="main"
    model_args.use_fast_tokenizer=True

    training_args = TrainingArguments(output_dir = "/sample/directory")
    training_args.output_dir = "/sample/directory"
    training_args.num_train_epochs = 1

    init_args = InitialArguments()
    init_args.results_logging_dir="/sample/directory"
    init_args.task_list="german_quad"


    assert isinstance(hgf_fine_tune_qa.run_task_evaluation(model_args, data_args, training_args, init_args), Trainer)
