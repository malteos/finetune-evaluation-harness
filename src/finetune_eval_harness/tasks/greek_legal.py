#from tasks.classification import Classification
from .classification import Classification


_DESCRIPTION = """
Dataset consisting of greek legal resources from Greek legislation
"""


_CITATION = """

@inproceedings{papaloukas-etal-2021-glc,
    title = "Multi-granular Legal Topic Classification on Greek Legislation",
    author = "Papaloukas, Christos and Chalkidis, Ilias and Athinaios, Konstantinos and Pantazi, Despina-Athanasia and Koubarakis, Manolis",
    booktitle = "Proceedings of the 3rd Natural Legal Language Processing (NLLP) Workshop",
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "",
    url = "https://arxiv.org/abs/2109.15298",
    doi = "",
    pages = ""
}

"""



class GreekLegal(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "greek_legal_code"  # HF datasets ID
    TASK_NAME = "greek_legal"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/greek_legal_code"

    
