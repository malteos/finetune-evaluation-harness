from abc import ABC, abstractmethod

class Classification(ABC):
    
    def get_task_type(self):
        return "classification"

    @abstractmethod
    def get_dataset_id(self):
        pass

    @abstractmethod
    def get_task_name(self):
        pass

    @abstractmethod
    def get_label_name(self):
        pass