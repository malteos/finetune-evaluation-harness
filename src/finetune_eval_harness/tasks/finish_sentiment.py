#from tasks.classification import Classification
from .classification import Classification


DESCRIPTION = """
Classification task for Finish sentiments 
"""


_CITATION = """

"""


class FinishSentiment(Classification):

    """
    Class for Finish Sentiment Classification
    """


    DATASET_ID = "sepidmnorozy/Finnish_sentiment"  # HF datasets ID
    TASK_NAME = "finish_sentiment"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Finnish_sentiment"

