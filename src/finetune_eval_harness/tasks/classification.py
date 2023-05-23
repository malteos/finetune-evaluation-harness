from abc import ABC, abstractmethod


class Classification(ABC):

    """
    Parent Class for defining Classification consisting of all 
    abstract methods which need to be implemented for each of
    the classification task. 
    """
    DATASET_ID = None
    TASK_NAME = None
    LABEL_NAME = None
    HOMEPAGE_URL = None
    DATASET_SPLIT = None
    PROBLEM_TYPE = None
    LANGUAGE = None

    def get_task_type(self):
        """
        return type
        """
        return "classification"

    
    def get_dataset_id(self):
        
        return self.DATASET_ID

    
    def get_task_name(self):
        
        return self.TASK_NAME

    
    def get_label_name(self):
        
        return self.LABEL_NAME

    
    def get_url(self):
        
        return self.HOMEPAGE_URL

    def get_dataset_split(self):
        
        return self.DATASET_SPLIT

    def get_problem_type(self):
        return self.PROBLEM_TYPE 
