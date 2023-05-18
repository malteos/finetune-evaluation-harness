# from tasks.ner import NamedEntityRecognition
from .classification import Classification
from .ner import NamedEntityRecognition


_DESCRIPTION = """
Adaption of E-health KD dataset challenge 
"""


_CITATION = """

@inproceedings{overview_ehealthkd2020,
  author    = {Piad{-}Morffis, Alejandro and
               Guti{\'{e}}rrez, Yoan and
               Ca{\~{n}}izares-Diaz, Hian and
               Estevez{-}Velarde, Suilan and 
               Almeida{-}Cruz, Yudivi{\'{a}}n and
               Mu{\~{n}}oz, Rafael and
               Montoyo, Andr{\'{e}}s},
  title     = {Overview of the eHealth Knowledge Discovery Challenge at IberLEF 2020},
  booktitle = ,
  year      = {2020},
}

"""


class EhealthKd(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """

    DATASET_ID = "ehealth_kd"  # HF datasets ID
    TASK_NAME = "ehealth_kd"
    HOMEPAGE_URL = "https://huggingface.co/datasets/fmmolina/eHealth-KD-Adaptation"
    LABEL_NAME = "entities"

