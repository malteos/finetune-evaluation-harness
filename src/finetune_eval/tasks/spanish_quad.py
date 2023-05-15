from tasks.qa import QuestionAnswering


class SpanishQuad(QuestionAnswering):

    """
    Class for German Quad Task
    """

    DATASET_ID = "squad_es"  # HF datasets ID
    TASK_NAME = "es_squad"
    HOMEPAGE_URL = "https://huggingface.co/datasets/squad_es"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME
    
    def get_url(self):
        return self.HOMEPAGE_URL
