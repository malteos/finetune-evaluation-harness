from tasks.ner import NamedEntityRecognition


class SzegedNer(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "ficsort/SzegedNER"  # HF datasets ID
    TASK_NAME = "szeged_ner"
    HOMEPAGE_URL = "https://huggingface.co/datasets/ficsort/SzegedNER"
    LABEL_NAME = "ner"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
    def get_label_name(self):
        return self.LABEL_NAME
    
