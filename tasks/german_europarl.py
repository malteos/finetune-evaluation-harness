import datasets 
import numpy as np 
#from . import TASK_REGISTRY, TASK_TYPE_REGISTRY, get_task
from tasks.ner import NamedEntityRecognition


class GermanEuroParl(NamedEntityRecognition):

    DATASET_ID = "akash418/german_europarl"    # HF datasets ID
    
    def get_dataset_id(self):
        return self.DATASET_ID
    
    