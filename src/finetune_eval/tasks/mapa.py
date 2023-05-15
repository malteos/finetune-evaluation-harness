from tasks.ner import NamedEntityRecognition


class Mapa(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "joelito/mapa"  # HF datasets ID
    TASK_NAME = "mapa"
    HOMEPAGE_URL = "https://huggingface.co/datasets/joelito/mapa"
    LABEL_NAME = "coarse_grained"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
    def get_label_name(self):
        return self.LABEL_NAME
    
