#from tasks.classification import Classification
from .classification import Classification


_DESCRIPTION = """
Topic classification for German News dataset
"""


_CITATION = """

@InProceedings{Schabus2017,
  Author    = {Dietmar Schabus and Marcin Skowron and Martin Trapp},
  Title     = {One Million Posts: A Data Set of German Online Discussions},
  Booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  Pages     = {1241--1244},
  Year      = {2017},
  Address   = {Tokyo, Japan},
  Doi       = {10.1145/3077136.3080711},
  Month     = aug
}

"""



class Gnad10(Classification):

    """
    Class for GNAD10 Classification Task
    """
    
    DATASET_ID = "gnad10"  # HF datasets ID
    TASKNAME = "gnad10"
    LABEL_NAME = "label"
    HOMEPAGE_URL = "https://huggingface.co/datasets/gnad10"
