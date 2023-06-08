import logging
import json
import os
import time

from datetime import datetime

from hf_scripts import hgf_fine_tune_class, utility_functions

from transformers import set_seed, Trainer, AutoTokenizer


logger = logging.getLogger(__name__)


class BaseTask(object):
    DATASET_ID = None
    TASK_NAME = None
    LABEL_NAME = None
    HOMEPAGE_URL = None
    DATASET_SPLIT = None
    PROBLEM_TYPE = None
    LANGUAGE = None

    tokenizer = None
    model = None
    config = None
    raw_datasets = None
    train_dataset = None
    eval_dataset = None
    label_column_name = None

    tokenized_train_dataset = None
    tokenized_eval_dataset = None

    def __init__(self, model_args, data_args, training_args, init_args) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.init_args = init_args

        self.data_args = self.add_labels_data_args(self.data_args)

    def get_task_type(self) -> str:
        raise NotImplementedError()

    def load_raw_datasets(self):
        raise NotImplementedError()

    def preprocess_datasets(self):
        raise NotImplementedError()

    def run_task_evaluation(self):
        raise NotImplementedError()

    def get_data_collator(self):
        raise NotImplementedError()

    def get_config(self):
        raise NotImplementedError()

    def get_model(self):
        raise NotImplementedError()

    def get_url(self):
        return self.HOMEPAGE_URL

    def get_label_name(self):
        return self.LABEL_NAME

    def get_task_name(self):
        return self.TASK_NAME

    def get_dataset_split(self):
        return self.DATASET_SPLIT

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_problem_type(self):
        return self.PROBLEM_TYPE

    def get_label_column_name(self):
        return None

    def get_train_dataset_name(self):
        return "train"

    def get_eval_dataset_name(self):
        return "test"

    def add_labels_data_args(self, data_args):
        data_args.label_value = self.get_label_name()
        data_args.dataset_config_name = self.get_dataset_split()
        data_args.dataset_name = self.get_dataset_id()

        if self.get_problem_type() is not None:
            data_args.special_task_type = self.get_problem_type()

        if self.get_task_type() == "ner":
            data_args.is_task_ner = True

        return data_args

    def get_max_seq_length(self):
        return self.data_args.max_seq_length  # self.get_tokenizer().model_max_length

    # def get_tokenizer(self):
    #     if self.tokenizer is None:
    #         self.tokenizer = utility_functions.load_tokenizer(
    #             self.model_args.tokenizer_name
    #             if self.model_args.tokenizer_name
    #             else self.model_args.model_name_or_path,
    #             self.model_args.cache_dir,
    #             self.model_args.use_fast_tokenizer,
    #             self.model_args.model_revision,
    #             self.model_args.use_auth_token,
    #             "left",
    #             None,
    #         )
    #         if self.tokenizer.pad_token is None:
    #             self.tokenizer.pad_token = self.tokenizer.eos_token

    #     return self.tokenizer

    def get_tokenizer(self):
        if self.tokenizer is None:
            tokenizer_name_or_path = (
                self.model_args.tokenizer_name
                if self.model_args.tokenizer_name
                else self.model_args.model_name_or_path
            )

            if self.get_config().model_type in {"bloom", "gpt2", "roberta"}:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path,
                    cache_dir=self.model_args.cache_dir,
                    use_fast=True,
                    revision=self.model_args.model_revision,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                    add_prefix_space=True,
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path,
                    cache_dir=self.model_args.cache_dir,
                    use_fast=True,
                    revision=self.model_args.model_revision,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                )

            if self.tokenizer.pad_token is None and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def get_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self.raw_datasets[self.get_train_dataset_name()]

            if self.data_args.max_train_samples is not None:
                max_train_samples = min(
                    len(self.train_dataset), self.data_args.max_train_samples
                )
                self.train_dataset = self.train_dataset.select(range(max_train_samples))

                logger.info(f"Train dataset limited to: {max_train_samples:,}")

        return self.train_dataset

    def get_eval_dataset(self):
        if self.eval_dataset is None:
            eval_dataset = self.raw_datasets[self.get_eval_dataset_name()]

            if self.data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(eval_dataset), self.data_args.max_eval_samples
                )
                eval_dataset = eval_dataset.select(range(max_eval_samples))

                logger.info(f"Eval dataset limited to: {max_eval_samples:,}")

            self.eval_dataset = eval_dataset

        return self.eval_dataset

    def get_trainer(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.get_data_collator(),
            compute_metrics=self.get_compute_metrics(),
        )

        return trainer

    def train(self):
        trainer = self.get_trainer()
        train_result = trainer.train()
        # metrics = train_result.metrics
        # metrics["train_samples"] = len(self.train_dataset)

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()

        return trainer, train_result

    def evaluate(self):
        # Set seed before initializing model.
        set_seed(self.training_args.seed)

        # Load raw datasets (download or from cache)
        self.raw_datasets = self.load_raw_datasets()

        # task.run_task_evaluation()
        self.get_config()
        self.get_tokenizer()
        self.get_model()

        # Dataset
        self.preprocess_datasets()

        trainer, train_result = self.train()

        # Evaluation metrics
        metrics = trainer.evaluate()

        model_name = self.model_args.model_name_or_path.split("/")[-1]

        metrics["train_metrics"] = train_result.metrics
        metrics["train_samples"] = len(self.train_dataset)
        metrics["eval_samples"] = len(self.eval_dataset)

        metrics["model_name"] = model_name
        metrics["task_name"] = self.TASK_NAME
        metrics["task_type"] = self.get_task_type()
        metrics["peft_choice"] = str(self.data_args.peft_choice)
        metrics["trainable_parameters_percentage"] = str(
            utility_functions.print_trainable_parameters(self.model)
        )
        metrics["training_args"] = self.training_args.to_dict()
        metrics["datetime"] = str(datetime.now())

        if self.LANGUAGE is not None:
            metrics["language"] = self.LANGUAGE

        if self.label_column_name is not None:
            metrics["label_column_name"] = self.label_column_name

        # Save metrics to disk
        if self.data_args.results_dir:
            if not os.path.exists(self.data_args.results_dir):
                os.makedirs(self.data_args.results_dir)

            output_path = os.path.join(
                self.data_args.results_dir,
                f"{int(time.time())}_{self.TASK_NAME}_{model_name}.json",
            )

            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info("Metrics saved to {output_path}")

        return metrics
