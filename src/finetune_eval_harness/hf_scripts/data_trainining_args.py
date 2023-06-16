from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Union

from dataclasses import asdict, dataclass, field, fields

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
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    max_seq_length: int = field(
        default=512,
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

    special_task_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Is this some special task (eg multi-label classification). Use: multi_label_classification"
        },
    )

    peft_choice: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which parameter efficent training strategy to use (LORA, P TUNING, PREFIX TUNING, PROMPT TUNING)"
        },
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Value of r for the parameter efficient fine-tuning"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Value of lora alpha for the parameter efficient fine-tuning"},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Value of lora dropout for the parameter efficient fine-tuning"},
    )
    lora_target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "Target modules. See https://github.com/huggingface/peft/blob/e48dfc331c6d06db1689c1d880ea05410b9a0ef5/src/peft/utils/other.py#L220"
        },
    )
    num_virtual_tokens: int = field(
        default=20,
        metadata={"help": "Number of virtual tokens to keep for parameter efficient fine-tuning"},
    )
    encoder_hidden_states: int = field(
        default=128,
        metadata={"help": "Encoder hidden state size to keep for parameter efficent fine-tuning"},
    )
    encoder_hidden_size: int = field(
        default=128,
        metadata={"help": "Encoder hidden state size to keep for parameter efficent fine-tuning"},
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
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    version_2_with_negative: bool = field(
        default=False,
        metadata={"help": "If true, some of the examples do not have an answer."},
    )
    feature_file: bool = field(default=False, metadata={"help": "Does your dataset has feature file on HF Hub"})
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
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
    results_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to store the results JSON files (to be used later for visualization)"},
    )
    base_checkpoint_dir: Optional[str] = field(
        default="", metadata={"help": "Path for storing model checkpoints and weights."}
    )
    doc_stride: int = field(
        default=16,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    is_subset: bool = field(default=False, metadata={"help": "Take subset of the datset"})

    bitfit: bool = field(default=False, metadata={"help": "Enable bias term trainig for base model."})

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d
