#from tasks.classification import Classification
from .classification import Classification


DESCRIPTION = """

WikiCAT_ca is a Spanish corpus for thematic Text Classification tasks. It is created automatically from Wikipedia and Wikidata sources, and contains 8401 articles from the Viquipedia classified under 12 different categories.

"""


_CITATION = """


"""



class WikiCatEs(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "PlanTL-GOB-ES/WikiCAT_esv2"  # HF datasets ID
    TASK_NAME = "wikicat_es"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/PlanTL-GOB-ES/WikiCAT_esv2"

    
