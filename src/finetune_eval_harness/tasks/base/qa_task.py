from hf_scripts import (
    hgf_fine_tune_qa,
    trainer_qa,
    utility_functions,
    utils_qa,
)
from tasks.base import BaseTask

import logging

import evaluate

from transformers import EvalPrediction
from datasets import load_dataset

logger = logging.getLogger(__name__)


class QuestionAnsweringTask(BaseTask):
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    def get_task_type(self) -> str:
        return "qa"

    def load_raw_datasets(self):
        self.raw_datasets = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config_name,
            cache_dir=self.model_args.cache_dir,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        return self.raw_datasets

    def get_config(self):
        return utility_functions.load_config(
            self.model_args.model_name_or_path,
            None,
            self.data_args.task_name,
            self.model_args.cache_dir,
            self.model_args.model_revision,
            self.model_args.use_auth_token,
            "qa",
            self.data_args,
        )

    def get_model(self):
        if self.model is None:
            model = utility_functions.load_model(
                self.model_args.model_name_or_path,
                bool(".ckpt" in self.model_args.model_name_or_path),
                self.get_config(),
                self.model_args.cache_dir,
                self.model_args.model_revision,
                self.model_args.use_auth_token,
                "qa",
            )

            model = utility_functions.freeze_layers(self.model_args, model)

            self.model = model

        return self.model

    def preprocess_datasets(self):
        column_names = self.raw_datasets[self.get_train_dataset_name()].column_names

        self.question_column_name = (
            "question" if "question" in column_names else column_names[0]
        )
        self.context_column_name = (
            "context" if "context" in column_names else column_names[1]
        )
        self.answer_column_name = (
            "answers" if "answers" in column_names else column_names[2]
        )

        pad_on_right = True

        # Create train feature from dataset
        fn_kwargs_train = {
            "data_args": self.data_args,
            "tokenizer": self.tokenizer,
            "question_column_name": self.question_column_name,
            "context_column_name": self.context_column_name,
            "max_seq_length": self.get_max_seq_length(),
            "pad_on_right": pad_on_right,
            "answer_column_name": self.answer_column_name,
        }

        with self.training_args.main_process_first(
            desc="train dataset map pre-processing"
        ):
            self.tokenized_train_dataset = self.get_train_dataset().map(
                utility_functions.prepare_train_features_qa,
                batched=True,
                fn_kwargs=fn_kwargs_train,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

        fn_kwargs_validation = {
            "tokenizer": self.tokenizer,
            "max_seq_length": self.get_max_seq_length(),
            "question_column_name": self.question_column_name,
            "context_column_name": self.context_column_name,
            "data_args": self.data_args,
            "pad_on_right": pad_on_right,
        }

        with self.training_args.main_process_first(
            desc="eval dataset map pre-processing"
        ):
            self.tokenized_eval_dataset = self.get_eval_dataset().map(
                utility_functions.prepare_features_validation_qa,
                batched=True,
                fn_kwargs=fn_kwargs_validation,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on eval dataset",
            )

    def get_data_collator(self):
        return utility_functions.data_collator_sequence_classification(
            self.data_args, self.training_args, self.tokenizer
        )

    # Post-processing:
    def post_processing_function(self, examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = utils_qa.postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.data_args.version_2_with_negative,
            n_best_size=self.data_args.n_best_size,
            max_answer_length=self.data_args.max_answer_length,
            null_score_diff_threshold=self.data_args.null_score_diff_threshold,
            output_dir=self.training_args.output_dir,
            log_level=self.training_args.get_process_log_level(),
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if self.data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        references = [
            {"id": ex["id"], "answers": ex[self.answer_column_name]} for ex in examples
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def get_trainer(self):
        metric = evaluate.load(
            "squad_v2" if self.data_args.version_2_with_negative else "squad"
        )

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        trainer = trainer_qa.QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_eval_dataset,
            eval_examples=self.eval_dataset,  # For matching with original text inputs
            tokenizer=self.tokenizer,
            data_collator=self.get_data_collator(),
            post_process_function=self.post_processing_function,
            compute_metrics=compute_metrics,
        )

        return trainer
