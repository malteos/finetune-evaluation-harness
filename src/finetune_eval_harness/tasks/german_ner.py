#from tasks.ner import NamedEntityRecognition
from .ner import NamedEntityRecognition

class GermanNerLegal(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "elenanereiss/german-ler"  # HF datasets ID
    TASK_NAME = "german_ner"
    HOMEPAGE_URL = "https://huggingface.co/datasets/elenanereiss/german-ler"

    
