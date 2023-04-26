import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from os import path
from datasets import load_dataset, ClassLabel, Value
import evaluate
from . import trainer_qa
from transformers import EvalPrediction, set_seed, HfArgumentParser, TrainingArguments
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# from utils_qa import postprocess_qa_predictions
from . import utils_qa
from hf_scripts.model_args import ModelArguments
from hf_scripts.data_trainining_args import DataTrainingArguments
from hf_scripts.utility_functions import *
from hf_scripts import utility_functions

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/question-answering/requirements.txt",
)


def run_task_evaluation(model_args, data_args, training_args, init_args):

    # model_args, data_args, training_args = parse_hf_arguments(args)
    (training_args, data_args) = utility_functions.prepend_data_args(
        training_args, data_args, init_args
    )
    send_example_telemetry("run_qa", model_args, data_args)
    logger = utility_functions.prepare_logger(training_args)

    last_checkpoint = utility_functions.detect_last_checkpoint(logger, training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = utility_functions.load_raw_dataset_qa(data_args, model_args)

    config = utility_functions.load_config(
        model_args.model_name_or_path,
        None,
        data_args.task_name,
        model_args.cache_dir,
        model_args.model_revision,
        model_args.use_auth_token,
        "qa",
    )
    tokenizer = utility_functions.load_tokenizer(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        model_args.cache_dir,
        model_args.use_fast_tokenizer,
        model_args.model_revision,
        model_args.use_auth_token,
        "left",
        None,
    )
    model = utility_functions.load_model(
        model_args.model_name_or_path,
        bool(".ckpt" in model_args.model_name_or_path),
        config,
        model_args.cache_dir,
        model_args.model_revision,
        model_args.use_auth_token,
        "qa",
    )

    model = utility_functions.freeze_layers(model_args, model)
    utility_functions.check_tokenizer_instance(tokenizer)

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.

    # this is imp step because the evaluation logic expects id of string
    print(raw_datasets["train"].features)
    new_features = raw_datasets["train"].features.copy()
    new_features["id"] = Value("string")
    raw_datasets["train"] = raw_datasets["train"].cast(new_features)
    raw_datasets["test"] = raw_datasets["test"].cast(new_features)

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    
    # no need of elif and else logic as train split will always have column names
    '''
    elif training_args.do_eval:
        # column_names = raw_datasets["validation"].column_names
        column_names = raw_datasets["test"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    '''


    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        fn_kwargs_train = {
            "data_args": data_args,
            "tokenizer": tokenizer,
            "question_column_name": question_column_name,
            "context_column_name": context_column_name,
            "max_seq_length": max_seq_length,
            "pad_on_right": pad_on_right,
            "answer_column_name": answer_column_name,
        }

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                utility_functions.prepare_train_features_qa,
                batched=True,
                fn_kwargs=fn_kwargs_train,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:

        eval_examples = raw_datasets["test"]
        # eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        fn_kwargs_validation = {
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
            "question_column_name": question_column_name,
            "context_column_name": context_column_name,
            "data_args": data_args,
            "pad_on_right": pad_on_right,
        }

        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_examples.map(
                utility_functions.prepare_features_validation_qa,
                batched=True,
                fn_kwargs=fn_kwargs_validation,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    predict_dataset = None
    predict_examples = None
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(
                range(data_args.max_predict_samples)
            )
        # Predict Feature Creation
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_examples.map(
                utility_functions.prepare_features_validation_qa,
                batched=True,
                fn_kwargs=fn_kwargs_validation,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    data_collator = utility_functions.data_collator_sequence_classification(
        data_args, training_args, tokenizer
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = utils_qa.postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=training_args.get_process_log_level(),
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    metrics_eval = utility_functions.train_eval_prediction(
        "question-answering",
        model,
        training_args,
        data_args,
        model_args,
        train_dataset,
        eval_dataset,
        eval_examples,
        data_collator,
        tokenizer,
        post_processing_function,
        compute_metrics,
        last_checkpoint,
        None,
        predict_dataset,
        predict_examples,
        None,
        False,
    )

    #trainer = utility_functions.set_hub_arguments(
    #    trainer, model_args, data_args, training_args, "question-answering"
    #)

    logger.info(f"Training Metrics {metrics_eval}")

    return metrics_eval


def main():
    run_task_evaluation()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # main()
    main(sys.argv[1:])
