# from tasks.classification import Classification
from .classification import Classification


_DESCRIPTION = """
Classification of social media data in dutch language (See more for data collection from the web page url) 
"""


_CITATION = """
@data{FK2/MTPTL7_2020, author = {Gupta, Aakash}, publisher = {COVID-19 Data Hub}, title = {{Dutch social media collection}}, year = {2020}, version = {DRAFT VERSION}, doi = {10.5072/FK2/MTPTL7}, url = {https://doi.org/10.5072/FK2/MTPTL7} }
"""


class DutchSocial(Classification):

    """
    Class for GermEval 2017 Classification Task
    """

    DATASET_ID = "dutch_social"  # HF datasets ID
    TASK_NAME = "dutch_social"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/dutch_social"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
