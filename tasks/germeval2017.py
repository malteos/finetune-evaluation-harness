import datasets
import numpy as np

# from . import TASK_REGISTRY, TASK_TYPE_REGISTRY, get_task
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

    DATASET_ID = "akash418/germeval_2017"  # HF datasets ID

    def get_dataset_id(self):
        return self.DATASET_ID
