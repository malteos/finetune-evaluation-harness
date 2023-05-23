
"""
Czech subjectivity dataset (Subj-CS) of 10k manually annotated subjective and objective sentences 
from movie reviews and descriptions. See the paper description https://arxiv.org/abs/2204.13915

Domain: Subjectivity Analysis

External URL's: https://github.com/pauli31/czech-subjectivity-dataset

"""

from .classification import Classification


_DESCRIPTION = """
Classification task for manually annotated subjective sentences 
"""


_CITATION = """
@article{pib2022czech,
    title={Czech Dataset for Cross-lingual Subjectivity Classification},
    author={Pavel Přibáň and Josef Steinberger},
    year={2022},
    eprint={2204.13915},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""



class CzechSubjectivity(Classification):

    DATASET_ID = "pauli31/czech-subjectivity-dataset"  # HF datasets ID
    TASK_NAME = "czech_subjectivity"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/pauli31/czech-subjectivity-dataset"
    LANGUAGE = "cs"

    
