import datasets
import numpy as np
from tasks.classification import Classification

# from . import TASK_REGISTRY, TASK_TYPE_REGISTRY, get_task


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

    DATASET_ID = "philschmid/germeval18"  # HF datasets ID

    def get_dataset_id(self):
        return self.DATASET_ID
    

