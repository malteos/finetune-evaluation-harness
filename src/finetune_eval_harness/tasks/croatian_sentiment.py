from .base.classification_task import ClassificationTask


_DESCRIPTION = """
Classification task for Croatian sentiment dataset
"""


_CITATION = """

"""


class CroatianSentiment(ClassificationTask):
    DATASET_ID = "sepidmnorozy/Croatian_sentiment"  # HF datasets ID
    TASK_NAME = "croatian_sentiment"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Croatian_sentiment"
    LANGUAGE = "hr"
