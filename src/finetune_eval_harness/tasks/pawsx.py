"""
PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification

The dataset consists of 23,659 human translated PAWS evaluation pairs and
296,406 machine translated training pairs in 6 typologically distinct languages.

Examples are adapted from  PAWS-Wiki

Git: https://github.com/google-research-datasets/paws/tree/master/pawsx
Paper: https://arxiv.org/abs/1908.11828

Prompt format (same as in mGPT):

"<s>" + sentence1 + ", right? " + mask + ", " + sentence2 + "</s>",

where mask is the string that matches the label:

Yes, No.

Example:

<s> The Tabaci River is a tributary of the River Leurda in Romania, right? No, The Leurda River is a tributary of the River Tabaci in Romania.</s>

Language specific prompts are translated word-by-word with Google Translate
and may differ from the ones used by mGPT and XGLM (they do not provide their prompts).

"""

from .classification import Classification

_CITATION = """
@inproceedings{yang-etal-2019-paws,
    title = "{PAWS}-{X}: A Cross-lingual Adversarial Dataset for Paraphrase Identification",
    author = "Yang, Yinfei  and
    Zhang, Yuan  and
    Tar, Chris  and
    Baldridge, Jason",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1382",
    doi = "10.18653/v1/D19-1382",
    pages = "3687--3692",
}"""


class PawsX(Classification):
    DATASET_ID = "paws-x"  # HF datasets ID
    TASK_NAME = "pawsx"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/paws-x"


class PawsXDe(PawsX):
    DATASET_SPLIT = "de"
    TASK_NAME = "pawsx_de"


class PawsXEn(PawsX):
    DATASET_SPLIT = "en"
    TASK_NAME = "pawsx_en"


class PawsXEs(PawsX):
    DATASET_SPLIT = "es"
    TASK_NAME = "pawsx_es"
