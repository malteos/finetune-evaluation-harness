from .base.classification_task import ClassificationTask

_DESCRIPTION = """
Classification task for Bulagariansentiment dataset
"""


_CITATION = """

"""


class BulgarianSentiment(ClassificationTask):
    DATASET_ID = "sepidmnorozy/Bulgarian_sentiment"
    TASK_NAME = "bulgarian_sentiment"
    LABEL_NAME = "label"
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Bulgarian_sentiment"
    LANGUAGE = "bg"
