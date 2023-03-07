from abc import ABC, abstractmethod

class Classification(ABC):
    
    def get_task_type(self):
        return "classification"

    @abstractmethod
    def get_dataset_id(self):
        pass