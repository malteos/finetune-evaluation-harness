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

    def get_task_type(self):
        return "qa"

    
    def get_dataset_id(self):
        return self.DATASET_ID
    
    @abstractmethod
    def get_url(self):
        pass
