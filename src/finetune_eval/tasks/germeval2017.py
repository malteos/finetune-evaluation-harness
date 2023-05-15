from tasks.classification import Classification

_CITATION = """
@inproceedings{germevaltask2017,
title = {{GermEval 2017: Shared Task on Aspect-based Sentiment in Social Media Customer Feedback}},
author = {Michael Wojatzki and Eugen Ruppert and Sarah Holschneider and Torsten Zesch and Chris Biemann},
year = {2017},
booktitle = {Proceedings of the GermEval 2017 - Shared Task on Aspect-based Sentiment in Social Media Customer Feedback},
address={Berlin, Germany},
pages={1--12}
}
"""


class GermEval2017(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "akash418/germeval_2017"  # HF datasets ID
    TASK_NAME = "germeval2017"
    LABEL_NAME = "relevance"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/malteos/germeval2017"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
