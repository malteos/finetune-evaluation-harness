
"""

WikiCAT_ca is a Spanish corpus for thematic Text Classification tasks. 
It is created automatically from Wikipedia and Wikidata sources, and contains 8401 articles 
from the Viquipedia classified under 12 different categories.

This dataset was developed by BSC TeMU as part of the PlanTL project, and intended as an 
evaluation of LT capabilities to generate useful synthetic corpus.

"""



from .classification import Classification


DESCRIPTION = """

WikiCAT_ca is a Spanish corpus for thematic Text Classification tasks. It is created automatically from Wikipedia and Wikidata sources, and contains 8401 articles from the Viquipedia classified under 12 different categories.

"""


_CITATION = """


"""



class WikiCatEs(Classification):


    DATASET_ID = "PlanTL-GOB-ES/WikiCAT_esv2"  
    TASK_NAME = "wikicat_es"
    LABEL_NAME = "label"  
    HOMEPAGE_URL = "https://huggingface.co/datasets/PlanTL-GOB-ES/WikiCAT_esv2"
    LANGUAGE = "es"
    
