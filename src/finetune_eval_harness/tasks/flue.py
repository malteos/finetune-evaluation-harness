#from tasks.classification import Classification
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

    DATASET_ID = "flue"  # HF datasets ID
    TASK_NAME = "flue"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/flue"

    
    