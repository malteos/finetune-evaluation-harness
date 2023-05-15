from tasks.classification import Classification

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

    """
    Class for GermEval 2018 Classification Task
    """

    DATASET_ID = "philschmid/germeval18"  # HF datasets ID
    TASK_NAME = "germeval2018"
    LABEL_NAME = "multi"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/philschmid/germeval18"

    def get_task_name(self):
        return self.TASK_NAME

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_label_name(self):
        return self.LABEL_NAME
    
    def get_url(self):
        return self.HOMEPAGE_URL
    