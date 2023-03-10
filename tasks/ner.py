from abc import ABC, abstractmethod


class NamedEntityRecognition(ABC):
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
