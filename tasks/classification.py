from abc import ABC, abstractmethod


class Classification(ABC):

    """
    Parent Class for defining Classification consisting of all 
    abstract methods which need to be implemented for each of
    the classification task. 
    """

    def get_task_type(self):
        """
        return type
        """
        return "classification"

    @abstractmethod
    def get_dataset_id(self):
        """
        implement method
        """
        pass

    @abstractmethod
    def get_task_name(self):
        """
        implement method
        """
        pass

    @abstractmethod
    def get_label_name(self):
        """
        implement method
        """
        pass
