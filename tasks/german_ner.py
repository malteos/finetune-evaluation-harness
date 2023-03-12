from tasks.ner import NamedEntityRecognition


class GermanNerLegal(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "elenanereiss/german-ler"  # HF datasets ID
    TASK_NAME = "german_ner"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME
