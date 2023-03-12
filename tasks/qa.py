from abc import ABC, abstractmethod

class QuestionAnswering(ABC):

    """
    Parent Class for defining Question Answering 
    consisting of all abstract methods which need to 
    be implemented for each of the QA task. 
    """


    def get_task_type(self):
        return "qa"

    @abstractmethod
    def get_dataset_id(self):
        pass
