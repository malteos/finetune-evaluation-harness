#from tasks.ner import NamedEntityRecognition
from .ner import NamedEntityRecognition


_DESCRIPTION = """

The dataset consists of 12 documents (9 for Spanish due to parsing errors) taken from EUR-Lex, a multilingual corpus of court decisions and legal dispositions in the 24 official languages of the European Union. 

"""


_CITATION = """

@article{DeGibertBonet2022,
author = {{de Gibert Bonet}, Ona and {Garc{\'{i}}a Pablos}, Aitor and Cuadros, Montse and Melero, Maite},
journal = {Proceedings of the Language Resources and Evaluation Conference},
number = {June},
pages = {3751--3760},
title = {{Spanish Datasets for Sensitive Entity Detection in the Legal Domain}},
url = {https://aclanthology.org/2022.lrec-1.400},
year = {2022}
}

"""


class Mapa(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "joelito/mapa"  # HF datasets ID
    TASK_NAME = "mapa"
    HOMEPAGE_URL = "https://huggingface.co/datasets/joelito/mapa"
    LABEL_NAME = "coarse_grained"

    
