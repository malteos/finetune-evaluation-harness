#from tasks.ner import NamedEntityRecognition
from .ner import NamedEntityRecognition

class SpanishEhealth(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "ehealth_kd"  # HF datasets ID
    TASK_NAME = "spanish_ehealth"
    HOMEPAGE_URL = "https://huggingface.co/datasets/ehealth_kd"

    
