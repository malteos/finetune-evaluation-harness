"""

Ten Thousand German News Articles Dataset
Paper of the original One Million Posts Corpus: https://dl.acm.org/doi/10.1145/3077136.3080711

Homepage: https://tblock.github.io/10kGNAD/
Git: https://github.com/tblock/10kGNAD

The 10k German News Article Dataset consists of 10273 German language news articles from the online Austrian newspaper website DER Standard.
Each news article has been classified into one of 9 categories by professional forum moderators employed by the newspaper.
This dataset is extended from the original One Million Posts Corpus. The dataset was created to support topic classification in German
because a classifier effective on a English dataset may not be as effective on a German dataset due to higher inflections and longer compound words.
Additionally, this dataset can be used as a benchmark dataset for German topic classification.

"""


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
    
    DATASET_ID = "gnad10"  
    TASKNAME = "gnad10"
    LABEL_NAME = "label"
    HOMEPAGE_URL = "https://huggingface.co/datasets/gnad10"
    LANGUAGE = "de"