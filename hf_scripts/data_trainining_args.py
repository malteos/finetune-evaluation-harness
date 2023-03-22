from dataclasses import dataclass, field
from typing import Optional, List

"""
Data Class for defining Data Training Arguments
"""

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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    train_batch_size: int = field(
        default=2, metadata={"help": "Size of train batch size"}
    )
    label_value: Optional[str] = field(
        default=None, metadata={"help": "label from the original dataset"}
    )
    remove_labels: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Labels which have to removed (please verify these from the original dataset)"
        },
    )
    peft_choice: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which parameter efficent training strategy to use (LORA, P TUNING, PREFIX TUNING, PROMPT TUNING)"
        },
    )
    r: int = field(
        default=8,
        metadata={"help": "Value of r for the parameter efficient fine-tuning"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={
            "help": "Value of lora alpha for the parameter efficient fine-tuning"
        },
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={
            "help": "Value of lora dropout for the parameter efficient fine-tuning"
        },
    )
    num_virtual_tokens: int = field(
        default=20,
        metadata={
            "help": "Number of virtual tokens to keep for parameter efficient fine-tuning"
        },
    )
    encoder_hidden_states: int = field(
        default=128,
        metadata={
            "help": "Encoder hidden state size to keep for parameter efficent fine-tuning"
        },
    )
    encoder_hidden_size: int = field(
        default=128,
        metadata={
            "help": "Encoder hidden state size to keep for parameter efficent fine-tuning"
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )
    epochs: int = field(default=1, metadata={"help": "Number of epochs"})
    save_after_steps: int = field(
        default=3500, metadata={"help": "Steps after which to save model checkpoints"}
    )
    version_2_with_negative: bool = field(
        default=False,
        metadata={"help": "If true, some of the examples do not have an answer."},
    )
    feature_file: bool = field(
        default=False, metadata={"help": "Does your dataset has feature file on HF Hub"}
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        },
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of text to input in the file (a csv or JSON file)."
        },
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of label to input in the file (a csv or JSON file)."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    is_task_ner: bool = field(default=False, metadata={"help": "Is the task NER"})
    results_log_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to store the results json file (to be used later for visualization)"
        },
    )
    base_checkpoint_dir: Optional[str] = field(
        default="", metadata={"help": "Path for storing model checkpoints and weights."}
    )
    doc_stride: int = field(
        default=16,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    '''
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.dataset_name is None:
            pass
        # elif self.train_file is None or self.validation_file is None:
        #    raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
    '''
