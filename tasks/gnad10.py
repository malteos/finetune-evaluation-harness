import datasets 
import numpy as np 
#from . import TASK_REGISTRY, TASK_TYPE_REGISTRY, get_task
from tasks.classification import Classification

class Gnad10():

    DATASET_ID = "gnad10"    # HF datasets ID

    def get_dataset_id(self):
        return self.DATASET_ID
    
    