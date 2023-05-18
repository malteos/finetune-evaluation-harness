#from tasks.classification import Classification
from .classification import Classification

_DESCRIPTION = """
Classification task for Bulagariansentiment dataset
"""


_CITATION = """

"""


class BulgarianSentiment(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "sepidmnorozy/Bulgarian_sentiment"  # HF datasets ID
    TASK_NAME = "bulgarian_sentiment"
    LABEL_NAME = "label"                            # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Bulgarian_sentiment"


    
