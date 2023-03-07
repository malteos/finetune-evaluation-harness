import os
from os import path
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from datasets import load_dataset
from hf_scripts.model_args import ModelArguments
from hf_scripts.data_trainining_args import DataTrainingArguments
from transformers import (
    PretrainedConfig,
    set_seed,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from hf_scripts.utility_functions import *
from typing import Any, Dict

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)


def main(args):
    model_args, data_args, training_args = parse_hf_arguments(args)
    send_example_telemetry("run_glue", model_args, data_args)
    logger = prepare_logger(training_args)
    last_checkpoint = detect_last_checkpoint(logger, training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_raw_dataset(data_args, training_args, model_args, logger)

    # Labels
    if data_args.label_value is not None:
        label_value = data_args.label_value
    elif "label" in raw_datasets.column_names:
        label_value = "label"
    else:
        label_value = str(raw_datasets.column_names["train"][-1])

    print("label_value", label_value)

    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            # label_list = raw_datasets["train"].features["label"].names
            label_list = raw_datasets["train"].features[label_value].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        is_regression = raw_datasets["train"].features[label_value].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # label_list = raw_datasets["train"].unique("label")
            label_list = raw_datasets["train"].unique(label_value)
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    all_columns = raw_datasets.column_names
    # column_others = all_columns['train'].remove(data_args.label_value)
    column_others = all_columns["train"].remove(label_value)

    config = load_config(
        model_args.model_name_or_path,
        num_labels,
        data_args.task_name,
        model_args.cache_dir,
        model_args.model_revision,
        model_args.use_auth_token,
        "sequence",
    )
    tokenizer = load_tokenizer(
        model_args.model_name_or_path,
        model_args.cache_dir,
        model_args.use_fast_tokenizer,
        model_args.model_revision,
        model_args.use_auth_token,
        "left",
        None,
    )
    model = load_model(
        model_args.model_name_or_path,
        bool(".ckpt" in model_args.model_name_or_path),
        config,
        model_args.cache_dir,
        model_args.model_revision,
        model_args.use_auth_token,
        "sequence",
    )

    print("parameters before", print_trainable_parameters(model))

    if data_args.peft_choice in peft_choice_list:
        model = load_model_peft(model, data_args, "SEQ_CLS")

    model = freeze_layers(model_args, model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    sentence1_key, sentence2_key = preprocess_raw_datasets(
        raw_datasets, data_args, label_value
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    fn_kwargs = {
        "tokenizer": tokenizer,
        "sentence1_key": sentence1_key,
        "sentence2_key": sentence2_key,
        "padding": padding,
        "max_seq_length": max_seq_length,
        "label_value": label_value,
        "label_to_id": label_to_id,
    }

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function_classification,
            batched=True,
            fn_kwargs=fn_kwargs,
            remove_columns=column_others,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" in raw_datasets:
            eval_dataset = raw_datasets["validation"]
        else:
            eval_dataset = raw_datasets["test"]

        # eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    predict_dataset = None
    if (
        training_args.do_predict
        or data_args.task_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    data_collator = data_collator_sequence_classification(
        data_args, training_args, tokenizer
    )

    # print(train_dataset[56])
    print("reamining trainable params", print_trainable_parameters(model))

    trainer = train_eval_prediction(
        "classification",
        model,
        training_args,
        data_args,
        model_args,
        train_dataset,
        eval_dataset,
        None,
        data_collator,
        tokenizer,
        None,
        compute_metrics_classification,
        last_checkpoint,
        label_value,
        predict_dataset,
        None,
        label_list,
        is_regression,
    )

    set_hub_arguments(
        trainer, model_args, data_args, training_args, "text-classification"
    )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # main()
    main(sys.argv[1:])
