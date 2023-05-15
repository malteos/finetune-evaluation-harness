from abc import ABC, abstractmethod


class NamedEntityRecognition(ABC):

    """
    Parent Class for defining Named Entity Recognition 
    consisting of all abstract methods which need to 
    be implemented for each of the NER task. 
    """

    DATASET_ID = None
    TASK_NAME = None
    LABEL_NAME = None

    def get_task_type(self):
        """
        return type
        """
        return "ner"

    
    def get_dataset_id(self):
        """
        implement 
        """
        return self.DATASET_ID

    @abstractmethod
    def get_task_name(self):
        """
        implement
        """
        pass

    @abstractmethod
    def get_url(self):
        """
        implement
        """
        pass 
