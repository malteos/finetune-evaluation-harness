from hf_scripts import utility_functions
from tasks.base import BaseTask, logger


from transformers import Trainer, set_seed


import itertools


class ClassificationTask(BaseTask):
    is_regression = False
    label_list = None
    num_labels = None
    label_column_name = None
    text_column_names = None
    label_to_id = None

    def get_label_column_name(self):
        if self.label_column_name is None:
            # determine
            if self.data_args.label_value is not None:
                self.label_column_name = self.data_args.label_value
            elif "label" in self.raw_datasets.column_names:
                self.label_column_name = "label"
            else:
                self.label_column_name = str(
                    self.raw_datasets.column_names["train"][-1]
                )

            logger.info(f"get_label_column_name {self.label_column_name}")

        return self.label_column_name

    def get_num_labels(self) -> int:
        if self.num_labels is None:
            # Labels not set -> automatically determine
            if self.is_regression:
                self.num_labels = 1
            else:
                self.num_labels = len(self.get_label_list())

        return self.num_labels

    def get_text_column_names(self):
        if self.text_column_names is None:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            # non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
            non_label_column_names = [
                name
                for name in self.raw_datasets["train"].column_names
                if name != self.get_label_column_name()
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

            self.text_column_names = sentence1_key, sentence2_key
        return self.text_column_names

    def get_label_list(self):
        if self.label_list is None:
            if isinstance(
                self.raw_datasets["train"][self.get_label_column_name()][0], list
            ):
                concat_list = list(
                    itertools.chain.from_iterable(
                        self.raw_datasets["train"][self.get_label_column_name()]
                    )
                )
                self.label_list = list(set(concat_list))
            else:
                self.label_list = self.raw_datasets["train"].unique(
                    self.get_label_column_name()
                )
                self.label_list.sort()  # Let's sort it for determinism

        return self.label_list

    def get_task_type(self) -> str:
        return "classification"

    def get_config(self):
        return utility_functions.load_config(
            self.model_args.model_name_or_path,
            self.get_num_labels(),
            self.data_args.task_name,
            self.model_args.cache_dir,
            self.model_args.model_revision,
            self.model_args.use_auth_token,
            "sequence",
            self.data_args,
        )

    def get_model(self):
        from transformers import PretrainedConfig

        if self.model is None:
            config = self.get_config()

            model = utility_functions.load_model(
                self.model_args.model_name_or_path,
                bool(".ckpt" in self.model_args.model_name_or_path),
                config,
                self.model_args.cache_dir,
                self.model_args.model_revision,
                self.model_args.use_auth_token,
                "sequence",
            )

            logger.info(
                f"parameters before {utility_functions.print_trainable_parameters(model)}"
            )

            if self.data_args.peft_choice in utility_functions.peft_choice_list:
                model = utility_functions.load_model_peft(
                    model, self.data_args, "SEQ_CLS"
                )

            model = utility_functions.freeze_layers(self.model_args, model)

            model.config.pad_token_id = model.config.eos_token_id
            model.resize_token_embeddings(len(self.get_tokenizer()))

            # Label index
            ######

            # Some models have set the order of the labels to use, so let's make sure we do use it.
            self.label_to_id = None

            if (
                model.config.label2id
                != PretrainedConfig(num_labels=self.get_num_labels()).label2id
                and self.data_args.task_name is not None
                and not self.is_regression
            ):
                # Some have all caps in their config, some don't.
                label_name_to_id = {
                    k.lower(): v for k, v in model.config.label2id.items()
                }
                if list(sorted(label_name_to_id.keys())) == list(
                    sorted(self.get_label_list())
                ):
                    self.label_to_id = {
                        i: int(label_name_to_id[self.get_label_list()[i]])
                        for i in range(self.get_num_labels())
                    }
                else:
                    logger.warning(
                        "Your model seems to have been trained with labels, but they don't match the dataset: ",
                        f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(self.get_label_list()))}."
                        "\nIgnoring the model labels as a result.",
                    )
            elif self.data_args.task_name is None and not self.is_regression:
                self.label_to_id = {v: i for i, v in enumerate(self.get_label_list())}

            if self.label_to_id is not None:
                model.config.label2id = self.label_to_id
                model.config.id2label = {
                    id: label for label, id in config.label2id.items()
                }
            elif self.data_args.task_name is not None and not self.is_regression:
                model.config.label2id = {
                    l: i for i, l in enumerate(self.get_label_list())
                }
                model.config.id2label = {
                    id: label for label, id in config.label2id.items()
                }

            # if data_args.max_seq_length > tokenizer.model_max_length:
            #     logger.warning(
            #         f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            #         f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            #     )

            self.model = model

        return self.model

    def load_raw_datasets(self):
        return utility_functions.load_raw_dataset(
            self.data_args, self.training_args, self.model_args, logger
        )

    def preprocess_datasets(self):
        padding = "max_length" if self.data_args.pad_to_max_length else False

        sentence1_key, sentence2_key = self.get_text_column_names()

        logger.info(f"{sentence1_key=}")
        logger.info(f"{sentence2_key=}")
        logger.info(f"{self.label_to_id=}")

        if hasattr(self, "filter_dataset"):
            logger.info("Filtering dataset ...")
            self.raw_datasets = self.raw_datasets.filter(self.filter_dataset)

        all_columns = self.raw_datasets.column_names
        # column_others = all_columns['train'].remove(data_args.label_value)
        column_others = all_columns["train"].remove(self.get_label_column_name())

        fn_kwargs = {
            "tokenizer": self.get_tokenizer(),
            "sentence1_key": sentence1_key,
            "sentence2_key": sentence2_key,
            "padding": padding,
            "max_seq_length": self.get_max_seq_length(),
            "label_value": self.get_label_column_name(),
            "label_to_id": self.label_to_id,
            "data_args": self.data_args,
        }

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            self.tokenized_train_dataset = self.get_train_dataset().map(
                utility_functions.preprocess_function_classification,
                batched=True,
                fn_kwargs=fn_kwargs,
                remove_columns=column_others,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            self.tokenized_eval_dataset = self.get_eval_dataset().map(
                utility_functions.preprocess_function_classification,
                batched=True,
                fn_kwargs=fn_kwargs,
                remove_columns=column_others,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on eval dataset",
            )

    def get_data_collator(self):
        return utility_functions.data_collator_sequence_classification(
            self.data_args, self.training_args, self.get_tokenizer()
        )

    def get_compute_metrics(self):
        return (
            utility_functions.compute_metrics_class_multi
            if self.data_args.special_task_type == "multi_label_classification"
            else utility_functions.compute_metrics_classification
        )

    def get_trainer(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,  # t if training_args.do_train else None,
            eval_dataset=self.tokenized_eval_dataset,  # if training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.get_data_collator(),
            compute_metrics=self.get_compute_metrics(),
        )

        return trainer
