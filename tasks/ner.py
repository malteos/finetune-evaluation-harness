from abc import ABC, abstractmethod


class NamedEntityRecognition(ABC):

    """
    Parent Class for defining Named Entity Recognition 
    consisting of all abstract methods which need to 
    be implemented for each of the NER task. 
    """

    def get_task_type(self):
        """
        return type
        """
        return "ner"

    @abstractmethod
    def get_dataset_id(self):
        """
        implement 
        """
        pass

    @abstractmethod
    def get_task_name(self):
        """
        implement
        """
        pass
