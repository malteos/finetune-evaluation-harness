#from tasks.classification import Classification
from .classification import Classification

_DESCRIPTION = """
Maltese version of PAWS-X dataset
"""


_CITATION = """


"""


class MalteseSentiment(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "amitness/PAWS-X-maltese"  # HF datasets ID
    TASK_NAME = "maltese_sentiment"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/amitness/PAWS-X-maltese"

    
