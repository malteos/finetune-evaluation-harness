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
import tempfile

"""
File consisting of integeration unit test cases for utility functions (implemented logic)
"""

#@pytest.mark.skip()
def test_add_labels_args():
    data_args = DataTrainingArguments()
    sample_task = "germeval2018"
    TASK_REGISTRY = {"germeval2018": germeval2018.GermEval2018}
    germeval_obj = germeval2018.GermEval2018()
    germeval_obj.get_dataset_id = MagicMock(return_value="philschmid/germeval18")
    germeval_obj.get_task_type = MagicMock(return_value="classification")
    germeval_obj.get_label_name = MagicMock(return_value="label")
    assert add_labels_data_args(sample_task, data_args) == data_args


#@pytest.mark.skip()
def test_prepend_data_args():
    data_args = DataTrainingArguments()
    temp_dir_name = tempfile.TemporaryDirectory().name
    training_args = TrainingArguments(output_dir= temp_dir_name)
    init_args = InitialArguments()
    #init_args.results_logging_dir = "/sample/directory"
    init_args.results_logging_dir = temp_dir_name
    #training_args.output_dir = "/sample/directory"
    training_args.output_dir = temp_dir_name
    assert prepend_data_args(training_args, data_args, init_args) == (
        training_args,
        data_args,
    )


@pytest.mark.skip()
def test_freeze():
    model_path = os.getcwd() + '/tests/custom_model'
    model_args = ModelArguments(model_name_or_path = model_path)
    model_args.freeze_layers = True
    model = AutoModelForTokenClassification.from_pretrained("bert-base-german-cased")
    assert freeze_layers(model_args, model) == model


#@pytest.mark.skip()
def test_load_config():
    temp_dir_name = tempfile.TemporaryDirectory().name
    model_path = os.getcwd() + '/tests/custom_model'
    #model_name_or_path = "bert-base-german-cased"
    model_name_or_path = model_path
    num_labels = 1
    finetuning_task = "classification"
    cache_dir = temp_dir_name
    #cache_dir = "/sample/directory"
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


#@pytest.mark.skip()
def test_load_model():
    model_path = os.getcwd() + '/tests/custom_model'
    temp_dir_name = tempfile.TemporaryDirectory().name
    #model_name_or_path = "bert-base-german-cased"
    model_name_or_path = model_path
    finetuning_task = "question-answering"
    cache_dir = temp_dir_name
    #cache_dir = "/sample/directory"
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


#@pytest.mark.skip()
def test_tasks_initialization():
    test_task = "german_ner_legal"
    assert isinstance(get_all_tasks(), list)


#@pytest.mark.skip()
def test_process_args():
    model_path = os.getcwd() + '/tests/custom_model'
    temp_dir_name = tempfile.TemporaryDirectory().name
    tasks_mock_obj = tasks
    tasks_mock_obj.get_all_tasks = MagicMock(return_value = ["germeval2018"])

    sample_cli_args = [
        "--model_name_or_path",
        #"bert-base-german-cased",
        model_path,
        "--tokenizer_name",
        "bert-base-german-cased",
        "--task_list",
        "germeval2018",
        "--base_checkpoint_dir",
        #"/sample/directory",
        temp_dir_name,
        "--results_logging_dir",
        #"/sample/directory",
        temp_dir_name,
        "--peft_choice",
        "lora",
        "--output_dir",
        #"/sample/directory",
        temp_dir_name,
    ]
    assert isinstance(process_arguments(sample_cli_args), HfArgumentParser)


#@pytest.mark.skip()
def test_get_labels():
    label_list = ["binary", "multi"]
    assert isinstance(get_label_list(label_list), list)


#@pytest.mark.skip()
def test_map_source_file():
    task_name = "gnad10"
    assert type(map_source_file(task_name))



#@pytest.mark.skip()
def test_init():
    germeval_obj = germeval2017.GermEval2017()
    assert isinstance(germeval_obj.get_url(), str)
    assert isinstance(german_europarl.GermanEuroParl().get_task_name(), str)
    assert isinstance(german_europarl.GermanEuroParl().get_task_type(), str)
    assert isinstance(german_europarl.GermanEuroParl().get_dataset_id(), str)
    assert isinstance(german_europarl.GermanEuroParl().get_url(), str)

    assert isinstance(german_ner.GermanNerLegal().get_url(), str)
    assert isinstance(german_ner.GermanNerLegal().get_task_type(), str)
    assert isinstance(german_ner.GermanNerLegal().get_task_name(), str)
    assert isinstance(german_ner.GermanNerLegal().get_dataset_id(), str)

    assert isinstance(german_quad.GermanQuad().get_dataset_id(), str)
    assert isinstance(german_quad.GermanQuad().get_task_type(), str)
    assert isinstance(german_quad.GermanQuad().get_task_name(), str)
    assert isinstance(german_quad.GermanQuad().get_url(), str)

    assert isinstance(gnad10.Gnad10().get_dataset_id(), str)
    assert isinstance(gnad10.Gnad10().get_task_type(), str)
    assert isinstance(gnad10.Gnad10().get_url(), str)
    assert isinstance(gnad10.Gnad10().get_task_name(), str)
    assert isinstance(gnad10.Gnad10().get_label_name(), str)

    assert isinstance(germeval2017.GermEval2017().get_dataset_id(), str)
    assert isinstance(germeval2017.GermEval2017().get_task_type(), str)
    assert isinstance(germeval2017.GermEval2017().get_url(), str)
    assert isinstance(germeval2017.GermEval2017().get_task_name(), str)
    assert isinstance(germeval2017.GermEval2017().get_label_name(), str)

    assert isinstance(germeval2018.GermEval2018().get_dataset_id(), str)
    assert isinstance(germeval2018.GermEval2018().get_task_type(), str)
    assert isinstance(germeval2018.GermEval2018().get_url(), str)
    assert isinstance(germeval2018.GermEval2018().get_task_name(), str)
    assert isinstance(germeval2018.GermEval2018().get_label_name(), str)

