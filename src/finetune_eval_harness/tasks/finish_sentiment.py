from .base.classification_task import ClassificationTask


DESCRIPTION = """
Classification task for Finish sentiments 
"""


_CITATION = """

"""


class FinishSentiment(ClassificationTask):

    """
    Class for Finish Sentiment Classification
    """

    DATASET_ID = "sepidmnorozy/Finnish_sentiment"  # HF datasets ID
    TASK_NAME = "finish_sentiment"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Finnish_sentiment"
    LANGUAGE = "fi"
