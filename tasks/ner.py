from abc import ABC, abstractmethod

class NamedEntityRecognition(ABC):
    def get_task_type(self):
        return "ner"

    @abstractmethod
    def get_dataset_id(self):
        pass