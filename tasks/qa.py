from abc import ABC, abstractmethod

class QuestionAnswering(ABC):
    def get_task_type(self):
        return "qa"

    @abstractmethod
    def get_dataset_id(self):
        pass
