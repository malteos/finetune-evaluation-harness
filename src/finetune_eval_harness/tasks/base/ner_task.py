from hf_scripts import hgf_fine_tune_ner, utility_functions
from tasks.base import BaseTask

from datasets import ClassLabel
from transformers import (
    DataCollatorForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    PretrainedConfig,
    Trainer,
)
from datasets import load_dataset
import evaluate
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NamedEntityRecognitionTask(BaseTask):
    text_column_name = None
    label_column_name = None
    label_list = None
    label_to_id = None
    num_labels = None
    b_to_i_label = None
    feature_file_exists = None
    labels_are_int = None

    def get_task_type(self) -> str:
        return "ner"

    def load_raw_datasets(self):
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config_name,
            cache_dir=self.model_args.cache_dir,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        column_names = raw_datasets[self.get_train_dataset_name()].column_names
        features = raw_datasets[self.get_train_dataset_name()].features

        if "tokens" in column_names:
            self.text_column_name = "tokens"
        else:
            self.text_column_name = column_names[0]

        if self.LABEL_NAME:
            label_column_name = self.LABEL_NAME
        elif self.data_args.is_task_ner:
            label_column_name = "ner_tags"
        else:
            label_column_name = column_names[1]

        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
        # Otherwise, we have to get the list of labels manually.
        labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
        if labels_are_int:
            label_list = features[label_column_name].feature.names
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = get_label_list(raw_datasets[self.get_train_dataset_name()][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}

        num_labels = len(label_list)

        self.label_list = label_list
        self.label_to_id = label_to_id
        self.num_labels = num_labels
        self.raw_datasets = raw_datasets
        self.label_column_name = label_column_name
        self.labels_are_int = labels_are_int

        logger.info(f"Label lists: {self.label_list}")

        return self.raw_datasets

    def get_config(self):
        if self.config is None:
            self.config = AutoConfig.from_pretrained(
                self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
                num_labels=self.num_labels,
                finetuning_task=self.data_args.task_name,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

        return self.config

    def get_model(self):
        if self.model is None:
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_args.model_name_or_path,
                config=self.config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
                ignore_mismatched_sizes=self.model_args.ignore_mismatched_sizes,
            )

            if self.model_args.resize_token_embeddings:
                logger.warning(f"Resizing token embeddings size to {len(self.get_tokenizer())}")
                model.resize_token_embeddings(len(self.get_tokenizer()))

            if self.data_args.peft_choice in utility_functions.peft_choice_list:
                model = utility_functions.load_model_peft(self.model, self.data_args, "TOKEN_CLS")

            model = utility_functions.freeze_layers(self.model_args, model)

            # Model has labels -> use them.
            if model.config.label2id != PretrainedConfig(num_labels=self.num_labels).label2id:
                if sorted(model.config.label2id.keys()) == sorted(self.label_list):
                    # Reorganize `label_list` to match the ordering of the model.
                    if self.labels_are_int:
                        self.label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(self.label_list)}
                        self.label_list = [model.config.id2label[i] for i in range(self.num_labels)]
                    else:
                        self.abel_list = [model.config.id2label[i] for i in range(self.num_labels)]
                        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
                else:
                    logger.warning(
                        "Your model seems to have been trained with labels, but they don't match the dataset: ",
                        f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
                        f" {sorted(self.label_list)}.\nIgnoring the model labels as a result.",
                    )

            # Set the correspondences label/ID inside the model config
            model.config.label2id = {l: i for i, l in enumerate(self.label_list)}
            model.config.id2label = dict(enumerate(self.label_list))

            # Map that sends B-Xxx label to its I-Xxx counterpart
            self.b_to_i_label = []
            for idx, label in enumerate(self.label_list):
                if label.startswith("B-") and label.replace("B-", "I-") in self.label_list:
                    self.b_to_i_label.append(self.label_list.index(label.replace("B-", "I-")))
                else:
                    self.b_to_i_label.append(idx)

            self.model = model

        return self.model

    def get_data_collator(self):
        return DataCollatorForTokenClassification(
            self.tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None
        )

    def preprocess_datasets(self):
        # Preprocessing the dataset
        # Padding strategy
        padding = "max_length" if self.data_args.pad_to_max_length else False

        # Tokenize all texts and align the labels with them.
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples[self.text_column_name],
                padding=padding,
                truncation=True,
                max_length=self.data_args.max_seq_length,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
            labels = []
            for i, label in enumerate(examples[self.label_column_name]):
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
                        label_ids.append(self.label_to_id[label[word_idx]])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        if self.data_args.label_all_tokens:
                            label_ids.append(self.b_to_i_label[self.label_to_id[label[word_idx]]])
                        else:
                            label_ids.append(-100)
                    previous_word_idx = word_idx

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        train_dataset = self.raw_datasets[self.get_train_dataset_name()]
        if self.data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        with self.training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

        eval_dataset = self.raw_datasets[self.get_eval_dataset_name()]

        if self.data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), self.data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        with self.training_args.main_process_first(desc="eval dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on eval dataset",
            )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # TODO remove unused columns
        self.tokenized_train_dataset = train_dataset
        self.tokenized_eval_dataset = eval_dataset

    def get_compute_metrics(self):
        # Metrics
        metric = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            if self.data_args.return_entity_level_metrics:
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

        return compute_metrics
