from .base.classification_task import ClassificationTask


DESCRIPTION = """

Classification dataset for slovak sentiment 

"""


_CITATION = """

"""


class SlovakSentiment(ClassificationTask):
    DATASET_ID = "sepidmnorozy/Slovak_sentiment"
    TASK_NAME = "slovak_sentiment"
    LABEL_NAME = "label"
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Slovak_sentiment"
    LANGUAGE = "sk"
