
"""

FLUE is an evaluation setup for French NLP systems similar to the popular GLUE benchmark. 
The goal is to enable further reproducible experiments in the future and to share models and progress on the French language. 
The tasks and data are obtained from existing works, please refer to our Flaubert paper for a complete list of references.


"""


from .classification import Classification


_DESCRIPTION = """
FLUE is an evaluation setup for French NLP systems similar to the popular GLUE benchmark
"""


_CITATION = """

@misc{le2019flaubert,
    title={FlauBERT: Unsupervised Language Model Pre-training for French},
    author={Hang Le and Loïc Vial and Jibril Frej and Vincent Segonne and Maximin Coavoux and Benjamin Lecouteux and Alexandre Allauzen and Benoît Crabbé and Laurent Besacier and Didier Schwab},
    year={2019},
    eprint={1912.05372},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

"""


class Flue(Classification):

    """
    Class for GermEval 2018 Classification Task
    """

    DATASET_ID = "flue"  
    TASK_NAME = "flue"
    LABEL_NAME = "label"  
    HOMEPAGE_URL = "https://huggingface.co/datasets/flue"
    LANGUAGE = "fr"
    
    