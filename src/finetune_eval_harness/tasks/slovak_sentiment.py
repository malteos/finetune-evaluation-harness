#from tasks.classification import Classification
from .classification import Classification


DESCRIPTION = """

Classification dataset for slovak sentiment 

"""


_CITATION = """



"""




class SlovakSentiment(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "sepidmnorozy/Slovak_sentiment"  # HF datasets ID
    TASK_NAME = "slovak_sentiment"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Slovak_sentiment"

    
