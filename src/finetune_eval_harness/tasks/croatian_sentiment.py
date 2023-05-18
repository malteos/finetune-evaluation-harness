#from tasks.classification import Classification
from .classification import Classification


_DESCRIPTION = """
Classification task for Croatian sentiment dataset
"""


_CITATION = """

"""

class CroatianSentiment(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "sepidmnorozy/Croatian_sentiment"  # HF datasets ID
    TASK_NAME = "croatian_sentiment"
    LABEL_NAME = "label"                            # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Croatian_sentiment"

    
