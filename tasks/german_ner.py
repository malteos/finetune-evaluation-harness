import datasets
import numpy as np
from tasks.ner import NamedEntityRecognition

# from . import TASK_REGISTRY, TASK_TYPE_REGISTRY, get_task


class GermanNerLegal(NamedEntityRecognition):

    DATASET_ID = "elenanereiss/german-ler"  # HF datasets ID

    def get_dataset_id(self):
        return self.DATASET_ID
