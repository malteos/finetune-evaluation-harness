import logging
import os
import sys
from dataclasses import dataclass, field
from hf_scripts.model_args import ModelArguments
from hf_scripts.data_trainining_args import DataTrainingArguments
from hf_scripts.initial_arguments import InitialArguments
import numpy as np
from . import trainer_qa
from hf_scripts import hgf_fine_tune_class, hgf_fine_tune_ner, hgf_fine_tune_qa
import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    get_peft_model,
    LoraConfig,
)
from peft import PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, TaskType
from peft.utils.other import fsdp_auto_wrap_policy
from os import path
from tasks import get_all_tasks, TASK_REGISTRY, TASK_TYPE_REGISTRY
import json
from typing import Any, Dict
from . import utils_qa
from datasets import load_dataset

"""
File containing utlility functions used over all the hf_scripts folder for all the tasks
"""


# dict used in sequence classification task
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

peft_choice_list = ["lora", "p_tune", "prefix_tune", "prompt_tune"]


def add_labels_data_args(each_task, data_args):
    """
    method to add labels and dataset name in data_args

    Args:
        each_task: str: type of the task
        data_args: an object of type DataTrainingArguments

    """

    dataset_name = TASK_REGISTRY[each_task]().get_dataset_id()
    if TASK_REGISTRY[each_task]().get_task_type() == "classification":
        label_value = TASK_REGISTRY[each_task]().get_label_name()
        data_args.label_value = label_value

    if TASK_REGISTRY[each_task]().get_task_type() == "ner":
        data_args.is_task_ner = True
    data_args.dataset_name = dataset_name

    return data_args


def prepend_data_args(
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    init_args: InitialArguments,
):
    """
    method to set necessary parameters for task evaluation

    Args:
        training_args: object of TrainingArguments
        data_args: object of DataTrainingArguments
        init_args: object of InitialArguments

    """

    data_args.results_log_path = init_args.results_logging_dir
    training_args.do_train = True
    training_args.do_eval = True
    training_args.overwrite_output_dir = True
    return (training_args, data_args)

'''
def parse_hf_arguments(args):
    """
    method to parse arguments in each of the hf script for each task

    Args:
        args: command line arguments passed while while running the main.py file

    Returns:
        triplet consisting of model_args, data_args and training_args

    """
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args=args
        )

    return model_args, data_args, training_args

'''
    

def freeze_layers(model_args: ModelArguments, model):
    """
    freeze layers of the model if parameter is passed

    Args:
        model_args: object of ModelArguments
        model: object of AutoModel

    """
    if model_args.freeze_layers == True:
        for param in model.base_model.parameters():
            param.requires_grad = False

    return model


def load_config(
    model_name_or_path: str,
    num_labels: int,
    finetuning_task: str,
    cache_dir: str,
    model_revision: str,
    use_auth_token: bool,
    model_type: str,
):
    """
    method for returning the model config type

    Args:
        model_name_or_path: name of the model according to hf hub
        num_labels: number of labels if the task is classification
        finetuning_task: the type of the task
        cache_dir: paramater of model_args class
        model_revision: parameter of model_args class
        use_auth_token: parameter of model_args class
        model_type: type of model

    Returns:
        config: object of AutoConfig

    """

    if model_type == "qa":
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            fine_tuning_task=finetuning_task,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=use_auth_token,
        )
        return config

    else:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            fine_tuning_task=finetuning_task,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=use_auth_token,
        )
        return config


def load_tokenizer(
    model_name_or_path: str,
    cache_dir: str,
    use_fast: bool,
    model_revision: str,
    use_auth_token: bool,
    padding_side: str,
    add_prefix_space: Any,
):
    """
    method for returning the tokenizer type

    Args:
        model_name_or_path: name of the model according to hf hub
        cache_dir: paramater of model_args class
        model_revision: parameter of model_args class
        use_auth_token: parameter of model_args class
        padding_side: padding side for the tokenizer
        add_prefix_space: adding prefix space for the tokenizerr

    Returns:
        tokenizer: object of AutoTokenizer

    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=use_fast,
        revision=model_revision,
        use_auth_token=use_auth_token,
        padding_side=padding_side if padding_side else "left",
        add_prefix_space=add_prefix_space if add_prefix_space else None,
    )
    return tokenizer


def load_model(
    model_name_or_path: str,
    from_tf: bool,
    config: AutoConfig,
    cache_dir: str,
    model_revision: str,
    use_auth_token: bool,
    model_type: str,
):
    """
    method for loading the model type

    Args:
        model_name_or_path: name of the model according to hf hub
        frmom_tf: bool object to load model from a checkpoint
        config: object of AutoConfig
        cache_dir: paramater of model_args
        model_revision: paramater of model_args
        use_auth_token: parameter of model_args
        model_type: type of the model

    Returns:
        model: AutoModel depending on various tasks (Classification, TokenClassification, and QuestionAnswering)

    """

    if model_type == "sequence":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=use_auth_token,
        )
    elif model_type == "ner":
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=use_auth_token,
        )
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=use_auth_token,
        )

    return model


def print_trainable_parameters(model: Any):
    """
    Prints the number of trainable parameters in the model.

    Args:
        model: object of AutoModel class

    Returns:
        float: percentage of parameters which are left trainable in the model

    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return 100 * trainable_params / all_param


def load_model_peft(
    model: Any,
    data_args: DataTrainingArguments,
    task_type: str,
):
    """
    method for returning the parameter efficent version of model

    Args:
        model: object of AutoModel class
        data_args: object of DataTrainingArgument class
        task_type: type of the task


    Returns:
        model: parameter efficent version of AutoModel class

    """

    task_type_dict = {"SEQ_CLS": TaskType.SEQ_CLS, "TOKEN_CLS": TaskType.TOKEN_CLS}

    if data_args.peft_choice == "lora":

        peft_config = LoraConfig(
            task_type=task_type_dict[task_type],
            inference_mode="False",
            # target_modules=["query", "value"],
            bias="none",
            r=data_args.r,
            lora_alpha=data_args.lora_alpha,
            lora_dropout=data_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        return model

    if data_args.peft_choice == "p_tune":
        peft_config = PromptEncoderConfig(
            task_type=task_type_dict[task_type],
            inference_mode="False",
            num_virtual_tokens=data_args.num_virtual_tokens,
            encoder_hidden_size=data_args.encoder_hidden_size,
        )
        model = get_peft_model(model, peft_config)
        return model

    if data_args.peft_choice == "prefix_tune":
        peft_config = PrefixTuningConfig(
            task_type=task_type_dict[task_type],
            num_virtual_tokens=data_args.num_virtual_tokens,
        )
        model = get_peft_model(model, peft_config)
        return model

    if data_args.peft_choice == "prompt_tune":
        peft_config = PromptTuningConfig(
            task_type=task_type_dict[task_type],
            num_virtual_tokens=data_args.num_virtual_tokens,
        )
        model = get_peft_model(model, peft_config)
        return model


def load_save_metrics_train(
    train_dataset, resume_from_checkpoint, trainer, last_checkpoint, max_train_samples
):
    """
    method for saving train metrics if do_train is True

    Args:
        train_dataset: train version of the HF dataset
        resume_from_checkpoint: resume incase the model training was stopped in between
        trainer: object of Trainer class
        last_checkpoint: detect and start training from the previous checkpoint
        max_train_samples: maximum of training samples

    Returns:
        trainer: object of Trainer class

    """

    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (
        max_train_samples if max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    return trainer


def load_save_metrics_validation(
    model_name_or_path: str,
    tasks: Any,
    trainer: Trainer,
    max_eval_samples: int,
    dataset_name: Any,
    problem_type: str,
    eval_datasets: list,
    peft_choice: str,
    remaining_params: float,
    problem_description: str,
    per_device_train_batch_size: int,
    label_value: str,
    results_log_path: str,
):

    """
    method for saving validation metrics if do_eval is True

    Args:
        model_name_or_path: name of the model according to HF hub
        trainer: object of the Trainer class
        max_eval_samples: maximum number of eval samples
        dataset_name: name of the dataset according to HF hub
        problem_type: type of the task
        eval_datasets: list of eval datasets
        peft_choice: choice of peft argument
        remaming_params: number of remaining parameters
        problem_description: type of problem
        per_device_train_batch_size: size of train batch size
        label_value: name of the label
        results_log_path: directory where the results should be logged

    """

    for eval_dataset in eval_datasets:
        metrics = trainer.evaluate()
        max_eval_samples = (
            max_eval_samples if max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        metrics["model_name"] = model_name_or_path
        metrics["dataset_name"] = dataset_name
        metrics["problem_type"] = problem_type
        metrics["peft_choice"] = str(peft_choice)
        metrics["trainable_parameters_percentage"] = str(remaining_params)
        metrics["problem_description"] = problem_description
        metrics["batch_size"] = per_device_train_batch_size
        if label_value is not None:
            metrics["label"] = label_value

        log_file_path = results_log_path + ".json"

        # check if file exists
        # if no then add the first entry
        if path.isfile(log_file_path) is False:
            with open(log_file_path, "w") as fp:
                metrics_list = []
                metrics_list.append(metrics)
                json.dump(metrics_list, fp)
            fp.close()

        else:
            # file exists read the prev entry, add new one and then write
            with open(log_file_path, "r") as new_file_path:
                curr_list = json.load(new_file_path)
                curr_list = curr_list + [metrics]

            with open(log_file_path, "w") as new_file_path:
                json.dump(curr_list, new_file_path)

            new_file_path.close()

        #trainer.log_metrics("eval", metrics)
        #trainer.save_metrics("eval", metrics)

        return metrics


def load_save_metrics_predict(
    trainer: Trainer,
    tasks: list,
    predict_datasets: Any,
    label_value: str,
    is_regression: bool,
    output_dir: str,
    label_list: list,
):

    """
    method for saving evaluation metrics incase do_predict = True

    Args:
        model_name_or_path: name of the model according to HF hub
        trainer: object of the Trainer class
        max_eval_samples: maximum number of eval samples
        dataset_name: name of the dataset according to HF hub
        problem_type: type of the task
        eval_datasets: list of eval datasets
        peft_choice: choice of peft argument
        remaming_params: number of remaining parameters
        problem_description: type of problem
        per_device_train_batch_size: size of train batch size
        label_value: name of the label
        results_log_path: directory where the results should be logged


    """

    for predict_dataset, task in zip(predict_datasets, tasks):
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        # predict_dataset = predict_dataset.remove_columns("label")
        predict_dataset = predict_dataset.remove_columns(label_value)
        predictions = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        ).predictions
        predictions = (
            np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
        )

        if(os.path.isdir(output_dir)):
            output_predict_file = os.path.join(output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def save_metrics_predict_ner(
    trainer: Trainer,
    training_args: TrainingArguments,
    predict_dataset: Any,
    output_dir: str,
    label_list: list,
):

    """
    method for saving prediction metrics for ner

    Args:
        model_name_or_path: name of the model according to HF hub
        trainer: object of the Trainer class
        max_eval_samples: maximum number of eval samples
        dataset_name: name of the dataset according to HF hub
        problem_type: type of the task
        eval_datasets: list of eval datasets
        peft_choice: choice of peft argument
        remaming_params: number of remaining parameters
        problem_description: type of problem
        per_device_train_batch_size: size of train batch size
        label_value: name of the label
        results_log_path: directory where the results should be logged


    """

    predictions, labels, metrics = trainer.predict(
        predict_dataset, metric_key_prefix="predict"
    )
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    # Save predictions
    output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_predictions_file, "w") as writer:
            for prediction in true_predictions:
                writer.write(" ".join(prediction) + "\n")


def save_metrics_predict_qa(data_args, trainer, predict_dataset, predict_examples):
    results = trainer.predict(predict_dataset, predict_examples)
    metrics = results.metrics

    max_predict_samples = (
        data_args.max_predict_samples
        if data_args.max_predict_samples is not None
        else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    return trainer


def preprocess_function_classification(
    examples: Dict[str, Any], **fn_kwargs
) -> Dict[str, Any]:
    """
    pre-process function for the classification task
    """
    sentence1_key = fn_kwargs["sentence1_key"]
    sentence2_key = fn_kwargs["sentence2_key"]
    tokenizer = fn_kwargs["tokenizer"]
    padding = fn_kwargs["padding"]
    max_seq_length = fn_kwargs["max_seq_length"]
    label_value = fn_kwargs["label_value"]
    label_to_id = fn_kwargs["label_to_id"]

    # Tokenize the texts
    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )

    result = tokenizer(
        *args, padding=padding, max_length=max_seq_length, truncation=True
    )
    # result = tokenizer(text = examples, padding = padding, max_length = max_seq_length, truncation = True)
    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and label_value in examples:
        result[label_value] = [
            (label_to_id[l] if l != -1 else -1) for l in examples[label_value]
        ]

    # if label_to_id is not None and "label" in examples:
    #    result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]

    result["labels"] = result[label_value].copy()

    return result


def compute_metrics_classification(p: EvalPrediction):

    """
    method for computing the classification metrics
    """

    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def detect_last_checkpoint(logger, training_args):
    """
    detect the last checkpoint

    Args:
        logger: object of Logger class
        training_args: object of TrainingArgument class

    """
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return last_checkpoint


def prepare_logger(training_args: TrainingArguments):
    """
    prepare the logger object

    Args:
        training_args: object of Training_Arguments

    Returns:
        logger: object of logger class

    """
    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    return logger


def data_collator_sequence_classification(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    tokenizer: AutoTokenizer,
):
    """
    define data collator for sequence classification

    Args:
        data_args: object of DataTrainingArguments
        training_args: object of TrainingArguments
        tokenizer: object of AutoTokenizer

    """
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    return data_collator


def set_hub_arguments(
    trainer: TrainingArguments,
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    task_name: str,
):
    """
    method for setting and uploading the model to the hub

    Args:
        trainer: object of Trainer class
        model_args: object of ModelTrainingArguments
        data_args; object of DataTrainingArguments
        training_args: object of TrainingArguments
        task_name: name of the task

    """

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": task_name}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return trainer


def load_raw_dataset(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    logger: Any,
):

    """
    method for handling the downloading of raw dataset

    Args:
        data_args: object of DataTrainingArguments
        training_args: object of TrainingArguments
        model_args: object of ModelArguments
        logger: object of Logger class

    """
    #raw_datasets = {}
    #raw_datasets["train"]=[]
    #raw_datasets["test"]=[]

    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    if data_args.is_subset == True:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:3%]",           # use 4% of the train set
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        raw_datasets["test"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"test[:1%]",           # use 1% of the test set
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    

    return raw_datasets


def load_raw_dataset_ner(data_args: DataTrainingArguments, model_args: ModelArguments):
    """
    download raw dataset

    Args:
        data_args: object of DataTrainingArguments
        training_args: object of TrainingArguments
        model_args: object of ModelArguments
        logger: object of Logger class

    """
    if data_args.is_subset == True:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:3%]",           # use 4% of the train set
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"validation[:1%]",           # use 1% of the test set
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # raw_datasets["test"] = load_dataset(
        #     data_args.dataset_name,
        #     data_args.dataset_config_name,
        #     split=f"validation[:1%]",           # use 1% of the test set
        #     cache_dir=model_args.cache_dir,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    '''
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension, data_files=data_files, cache_dir=model_args.cache_dir
        )
    '''

    return raw_datasets

def load_raw_dataset_qa(data_args: DataTrainingArguments, model_args: ModelArguments):

    if data_args.is_subset == True:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:3%]",           # use 4% of the train set
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        raw_datasets["test"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"test[:1%]",           # use 1% of the test set
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        
    return raw_datasets



def preprocess_raw_datasets(
    raw_datasets: Any, data_args: DataTrainingArguments, label_value: str
):

    """
    method for pre-processing raw datasets

    Args:
        data_args: object of DataTrainingArguments
        training_args: object of TrainingArguments
        model_args: object of ModelArguments
        logger: object of Logger class


    """

    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        # non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != label_value
        ]
        # print("non_label_columns", non_label_column_names)
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    return sentence1_key, sentence2_key


def get_label_list(labels: list):
    """
    return label list for ner task

    Args:
        labels: list of the labels

    """
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def tokenize_and_align_labels(examples: Dict[str, Any], **fn_kwargs) -> Dict[str, Any]:

    """
    returns object of Tokenizer class

    Args:
        examples: samples from the dataset
        fn_kwargs: other columns relating to aligning labels for ner

    Returns:
        tokenized_inputs: object of Tokenizer class

    """

    text_column_name = fn_kwargs["text_column_name"]
    padding = fn_kwargs["padding"]
    data_args = fn_kwargs["data_args"]
    tokenizer = fn_kwargs["tokenizer"]
    label_column_name = fn_kwargs["label_column_name"]
    label_to_id = fn_kwargs["label_to_id"]
    b_to_i_label = fn_kwargs["b_to_i_label"]

    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=data_args.max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if data_args.label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def check_tokenizer_instance(tokenizer: AutoTokenizer):

    """
    check if tokenizer is PretrainedTokenizer

    Args:
        tokenizer: object of AutoTokenizer

    Returns:
        raises an excpetion if its not the fast version of the Tokenizer

    """

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )


def generate_b_to_i_label(feature_file_exists: bool, label_list: list):

    """
    method to generate b_to_i_label

    Args:
        feature_file_exists: boolean value indicating if feature file is there in the hf hub
        label_list: list of labels from the hf dataset

    Returns:
        list mapping B- to I- labels if it exists

    """

    b_to_i_label = []
    if feature_file_exists:
        for idx, label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

    for idx, label in enumerate(label_list):
        b_to_i_label.append(idx)

    return b_to_i_label


def map_train_validation_predict_ds_ner(
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    raw_datasets: Any,
    fn_kwargs: Any,
):

    """
    logic for mapping train, test and validation splits for the ner task

    Args:
        training_args: object of TrainingArguments
        data_args: object of DataTrainingArguments
        raw_datasets: datasets object from the hf
        fn_kwargs: dict containing other relevant paramaters for the operation

    Returns:
        train, evaluation and prediction splits of the dataset


    """

    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                fn_kwargs=fn_kwargs,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                fn_kwargs=fn_kwargs,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            
    return train_dataset, eval_dataset, predict_dataset


def generate_label_list(
    raw_datasets: Any, features: Any, ClassLabel: Any, label_column_name: str
):
    """
    method to generate label_list and label_to_id

    Args:
        raw_datasets: datasets from the hf hub
        features: an object of type Feature, see hf datasets
        ClassLabel:

    """

    if isinstance(features[label_column_name].feature, ClassLabel) == True:
        feature_file_exists = True
    else:
        feature_file_exists = False

    if feature_file_exists:
        labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    else:
        labels_are_int = 0

    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    return label_list, label_to_id, feature_file_exists, labels_are_int


def prepare_train_features_qa(
    examples: Dict[str, Any], **fn_kwargs_train
) -> Dict[str, Any]:

    """
    method to prepare train features for qa task

    Args:
        examples: instances from the processed dataset
        fn_kwargs: dict containing other fields for preparing train features

    Returns:
        tokenized_examples: dict containing tokenized examples

    """

    question_column_name = fn_kwargs_train["question_column_name"]
    data_args = fn_kwargs_train["data_args"]
    tokenizer = fn_kwargs_train["tokenizer"]
    context_column_name = fn_kwargs_train["context_column_name"]
    max_seq_length = fn_kwargs_train["max_seq_length"]
    pad_on_right = fn_kwargs_train["pad_on_right"]
    answer_column_name = fn_kwargs_train["answer_column_name"]

    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [
        q.lstrip() for q in examples[question_column_name]
    ]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_features_validation_qa(
    examples: Dict[str, Any], **fn_kwargs_validation
) -> Dict[str, Any]:

    """
    method to generate features for validation dataset for qa task

    Args:
        examples: instances from processed validation dataset
        fn_kwargs_validation: dict containing other fields for processing validation dataset

    Returns:
        tokenized_examples: dict containing tokenized examples

    """

    question_column_name = fn_kwargs_validation["question_column_name"]
    tokenizer = fn_kwargs_validation["tokenizer"]
    context_column_name = fn_kwargs_validation["context_column_name"]
    pad_on_right = fn_kwargs_validation["pad_on_right"]
    max_seq_length = fn_kwargs_validation["max_seq_length"]
    data_args = fn_kwargs_validation["data_args"]

    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [
        q.lstrip() for q in examples[question_column_name]
    ]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        # print(examples["id"])
        # examples_id_str = [str(j) for j in examples["id"]]
        # print(examples_id_str)

        tokenized_examples["example_id"].append(examples["id"][sample_index])
        # tokenized_examples["example_id"].append(examples_id_str[sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

    
def train_eval_prediction(
    task_type: str,
    model: Any,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    train_dataset: Any,
    eval_dataset: Any,
    eval_examples: Any,
    data_collator: Any,
    tokenizer: AutoTokenizer,
    post_processing_function: Any,
    compute_metrics: Any,
    last_checkpoint: Any,
    label_value: str,
    predict_dataset: Any,
    predict_examples: Any,
    label_list: str,
    is_regression: bool,
):

    """
    method consisting of training, evaluation and prediction loop logic for each of the task

    Args:
        task_type: type of the task
        model: object of AutoModel
        training_args: object of TrainingArguments
        data_args: object of DataTrainingArgument
        model_args: object of ModelArguments
        train_dataset: train dataset of HF dataset
        eval_dataset: eval dataset of HF dataset
        eval_example: examples from the evaluation dataset
        data_collator: data collator function
        tokenizer: object of AutoTokenizer
        post_processing_function: post processing function
        compute_metrics: metric of seqeval
        last_checkpoint: last checkpoint during training
        label_value: name of the label
        predict_dataset: prediction dataset
        predict_examples: examples from the prediction dataset
        label_list: list of labels
        is_regression: is the task conccerning regression

    """

    if task_type == "question-answering":
        trainer = trainer_qa.QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    if training_args.do_train:
        trainer = load_save_metrics_train(
            train_dataset,
            training_args.resume_from_checkpoint,
            trainer,
            last_checkpoint,
            data_args.max_train_samples,
        )

    if training_args.do_eval:
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        metrics_eval = load_save_metrics_validation(
            model_args.model_name_or_path,
            tasks,
            trainer,
            data_args.max_eval_samples,
            data_args.dataset_name,
            task_type,
            eval_datasets,
            data_args.peft_choice,
            print_trainable_parameters(model),
            model.config.problem_type,
            training_args.per_device_train_batch_size,
            label_value,
            data_args.results_log_path,
        )

    if training_args.do_predict:
        tasks = [data_args.task_name]
        if task_type == "classification" or task_type == "question-answering":
            predict_datasets = [predict_dataset]

        if task_type == "question_answering":
            trainer = save_metrics_predict_qa(
                data_args, trainer, predict_dataset, predict_examples
            )
        if task_type == "classification":
            load_save_metrics_predict(
                trainer,
                tasks,
                predict_datasets,
                label_value,
                is_regression,
                training_args.output_dir,
                label_list,
            )
        if task_type == "token-classification":
            save_metrics_predict_ner(
                trainer,
                training_args,
                predict_dataset,
                training_args.output_dir,
                label_list,
            )

    return metrics_eval


def map_source_file(task_name: str):

    """
    identify the task_type and return the name of the appropriate hf script name

    Args:
        task_name: name of the task (classification, ner and qa)

    """
    task_type = TASK_TYPE_REGISTRY[task_name]
    if task_type == "classification":
        return hgf_fine_tune_class
    if task_type == "ner":
        return hgf_fine_tune_ner
    else:
        return hgf_fine_tune_qa


def get_true_predictions_labels(
    label_list: list,
    predictions: Any,
    labels: list,
    metric: Any,
    data_args: DataTrainingArguments,
):
    """
    get true predictions and labels for ner task

    Args:
        label_list: list of labels
        predictins: prediction tensor from the model
        labels: list of labels
        metric: seqeval metric
        data_args: object of DataTrainingArguments

    """
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    if data_args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
