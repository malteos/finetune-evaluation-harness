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
    HOMEPAGE_URL = None
    DATASET_SPLIT = None
    LANGUAGE = None

    def get_task_type(self):
        """
        return type
        """
        return "ner"

    
    def get_dataset_id(self):
        
        return self.DATASET_ID

    #@abstractmethod
    def get_task_name(self):
        
        return self.TASK_NAME

    #@abstractmethod
    def get_url(self):
        
        return self.HOMEPAGE_URL
    
    def get_dataset_split(self):

        return self.DATASET_SPLIT

    def get_label_name(self):
        
        return self.LABEL_NAME
