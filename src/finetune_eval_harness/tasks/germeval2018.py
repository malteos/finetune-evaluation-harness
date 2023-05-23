"""
Overview of the GermEval 2018 Shared Task on the Identification of Offensive Language
https://epub.oeaw.ac.at/0xc1aa5576_0x003a10d2.pdf

The GermEval2018 task is a benchmark to indetify offensive language. This shared
task deals with the classification of German tweets from Twitter. It comprises two tasks,
a coarse-grained binary classification task (OTHER, OFFENSE) and a fine-grained multi-class classification
task(OTHER, INSULT, ABUSE, PROFANITY). This script focuses on the binary (coarse) classification.

Homepage: https://projects.cai.fbi.h-da.de/iggsa/

"""


from .classification import Classification

_DESCRIPTION = """

"""


_CITATION = """
@inproceedings{vamvas2020germeval,
    author    = "Wiegand, Michael, and Siegel, Melanie and Ruppenhofer, Josef",
    title     = "Overview of the GermEval 2018 Shared Task on the Identification of Offensive Language",
    booktitle = "Proceedings of the GermEval 2018 Workshop  14th Conference on Natural Language Processing (KONSENS)",
    address   = "Vienna, SAustria",
    year      = "2018",
    month     = "sep",
    url       = "https://epub.oeaw.ac.at/0xc1aa5576_0x003a10d2.pdf"
}"""


class GermEval2018(Classification):


    DATASET_ID = "philschmid/germeval18"    
    TASK_NAME = "germeval2018"
    LABEL_NAME = "multi"                    
    HOMEPAGE_URL = "https://huggingface.co/datasets/philschmid/germeval18"
    LANGUAGE = "de"
