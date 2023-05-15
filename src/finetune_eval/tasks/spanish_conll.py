from tasks.ner import NamedEntityRecognition


class SpanishConll(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "PlanTL-GOB-ES/CoNLL-NERC-es"  # HF datasets ID
    TASK_NAME = "spanish_conll"
    HOMEPAGE_URL = "https://huggingface.co/datasets/PlanTL-GOB-ES/CoNLL-NERC-es"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
