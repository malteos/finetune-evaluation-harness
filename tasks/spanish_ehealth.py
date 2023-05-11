from tasks.ner import NamedEntityRecognition


class SpanishEhealth(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "ehealth_kd"  # HF datasets ID
    TASK_NAME = "spanish_ehealth"
    HOMEPAGE_URL = "https://huggingface.co/datasets/ehealth_kd"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
