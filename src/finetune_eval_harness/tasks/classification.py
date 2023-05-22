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

    def get_task_type(self):
        """
        return type
        """
        return "classification"

    
    def get_dataset_id(self):
        """
        implement method
        """
        return self.DATASET_ID

    
    def get_task_name(self):
        """
        implement method
        """
        return self.TASK_NAME

    
    def get_label_name(self):
        """
        implement method
        """
        return self.LABEL_NAME

    
    def get_url(self):
        """
        implement method
        """
        return self.HOMEPAGE_URL

    def get_dataset_split(self):
        """
        implement method
        """
        return self.DATASET_SPLIT

    def get_problem_type(self):
        return self.PROBLEM_TYPE 
