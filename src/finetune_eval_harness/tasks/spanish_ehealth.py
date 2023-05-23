"""

Dataset of the eHealth-KD Challenge at IberLEF 2020. 
It is designed for the identification of semantic entities and relations in Spanish health documents.

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

from .ner import NamedEntityRecognition

class SpanishEhealth(NamedEntityRecognition):

    
    DATASET_ID = "ehealth_kd"  
    TASK_NAME = "spanish_ehealth"
    HOMEPAGE_URL = "https://huggingface.co/datasets/ehealth_kd"
    LANGUAGE = "es"
    
