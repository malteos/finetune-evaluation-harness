"""

GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

Available tasks: glue_cola,glue_sst2,glue_stsb,glue_wnli

"""
from .base.classification_task import ClassificationTask

_CITATION = """

@inproceedings{wang2019glue,
  title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
  note={In the Proceedings of ICLR.},
  year={2019}
}

"""


class GlueBaseTask(ClassificationTask):
    DATASET_ID = "glue"
    HOMEPAGE_URL = "https://huggingface.co/datasets/glue"
    LANGUAGE = "en"

    def get_eval_dataset_name(self):
        return "validation"


class GlueColaTask(GlueBaseTask):
    DATASET_SPLIT = "cola"
    TASK_NAME = "glue_cola"

    text_column_names = "sentence", None
    label_column_name = "label"


class GlueSST2Task(GlueBaseTask):
    DATASET_SPLIT = "sst2"
    TASK_NAME = "glue_sst2"

    text_column_names = "sentence", None
    label_column_name = "label"


class GlueSTSBTask(GlueBaseTask):
    DATASET_SPLIT = "stsb"
    TASK_NAME = "glue_stsb"

    text_column_names = "sentence1", "sentence2"
    label_column_name = "label"
    PROBLEM_TYPE = "regression"


class GlueWNLITask(GlueBaseTask):
    DATASET_SPLIT = "wnli"
    TASK_NAME = "glue_wnli"

    text_column_names = "sentence1", "sentence2"
    label_column_name = "label"
