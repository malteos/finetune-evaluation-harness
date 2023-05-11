from tasks.ner import NamedEntityRecognition


class EhealthKd(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "ehealth_kd"  # HF datasets ID
    TASK_NAME = "ehealth_kd"
    HOMEPAGE_URL = "https://huggingface.co/datasets/fmmolina/eHealth-KD-Adaptation"
    LABEL_NAME = "entities"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
    def get_label_name(self):
        return self.LABEL_NAME
    
