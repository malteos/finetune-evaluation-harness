from abc import ABC, abstractmethod

class QuestionAnswering(ABC):

    """
    Parent Class for defining Question Answering 
    consisting of all abstract methods which need to 
    be implemented for each of the QA task. 
    """

    DATASET_ID = None
    TASK_NAME = None
    LABEL_NAME = None
    HOMEPAGE_URL = None
    LANGUAGE = None
    DATASET_SPLIT = None

    def get_task_type(self):
        return "qa"

    
    def get_dataset_id(self):
        return self.DATASET_ID
    
    #@abstractmethod
    def get_url(self):
        return self.HOMEPAGE_URL
    
    def get_label_name(self):
        return self.LABEL_NAME
    
    def get_task_name(self):
        return self.TASK_NAME
    
    def get_dataset_split(self):
        return self.DATASET_SPLIT
