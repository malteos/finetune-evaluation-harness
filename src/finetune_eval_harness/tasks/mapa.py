"""

The dataset consists of 12 documents (9 for Spanish due to parsing errors) taken from EUR-Lex, 
a multilingual corpus of court decisions and legal dispositions in the 24 official languages of the European Union. 
The documents have been annotated for named entities following the guidelines of the MAPA project which foresees two annotation level, 
a general and a more fine-grained one. The annotated corpus can be used for named entity recognition/classification.


"""



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

    
    DATASET_ID = "joelito/mapa"  
    TASK_NAME = "mapa"
    HOMEPAGE_URL = "https://huggingface.co/datasets/joelito/mapa"
    LABEL_NAME = "coarse_grained"


class MapaDe(Mapa):

    DATASET_SPLIT = "de"
    TASK_NAME = "mapa_de"

class MapaFr(Mapa):

    DATASET_SPLIT = "fr"
    TASK_NAME = "mapa_fr"

    
class MapaEn(Mapa):

    DATASET_SPLIT = "en"
    TASK_NAME = "mapa_en"