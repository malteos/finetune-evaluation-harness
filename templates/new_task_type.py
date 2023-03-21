from abc import ABC, abstractmethod

class NewTaskType(ABC):
    """
    template class for defining a new task type different from exisitng ones in /tasks folder
    """

    # add HF hub dataset ID
    DATASET_ID = None

    # Specify a unique name of the task
    TASK_NAME = None

    # Does it have a label? Uf yes please specify
    LABEL_NAME = None

    """
    Next specify the list of abstract methods which will be implemented sequentially
    If you add some new fields, create getter methods for that
    """

    def get_task_type(self):
        """
        return type (string identifying the type of task e.g classification, named entity recognition (ner), etc)
        """
        return ""

    
    def get_dataset_id(self):
        """
        implement method
        """
        return self.DATASET_ID

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

    @abstractmethod
    def get_url(self):
        """
        implement method
        """
        pass

