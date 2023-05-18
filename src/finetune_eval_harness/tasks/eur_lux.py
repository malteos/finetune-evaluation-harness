#from tasks.classification import Classification
#from tasks.ner import NamedEntityRecognition
from .classification import Classification



_DESCRIPTION = """
Classification dataset for 8 different concepts spanning across multiple languages (please refer to url for more details)
"""


_CITATION = """

@InProceedings{chalkidis-etal-2021-multieurlex,
  author = {Chalkidis, Ilias  
                and Fergadiotis, Manos
                and Androutsopoulos, Ion},
  title = {MultiEURLEX -- A multi-lingual and multi-label legal document 
               classification dataset for zero-shot cross-lingual transfer},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods
               in Natural Language Processing},
  year = {2021},
  publisher = {Association for Computational Linguistics},
  location = {Punta Cana, Dominican Republic},
  url = {https://arxiv.org/abs/2109.00904}
}

"""





class EurLux(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "multi_eurlex"  # HF datasets ID
    TASK_NAME = "eur_lux"
    LABEL_NAME = "labels"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/multi_eurlex"


    
