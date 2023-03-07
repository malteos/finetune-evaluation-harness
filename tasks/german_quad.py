import datasets
import numpy as np
from tasks.qa import QuestionAnswering
# from . import TASK_REGISTRY, TASK_TYPE_REGISTRY, get_task


class GermanQuad(QuestionAnswering):

    DATASET_ID = "deepset/germanquad"  # HF datasets ID
    
    def get_dataset_id(self):
        return self.DATASET_ID

    
