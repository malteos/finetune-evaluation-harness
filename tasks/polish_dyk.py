from tasks.qa import QuestionAnswering


class PolishDyk(QuestionAnswering):

    """
    Class for German Quad Task
    """

    DATASET_ID = "allegro/klej-dyk"  # HF datasets ID
    TASK_NAME = "polish_dyk"
    HOMEPAGE_URL = "https://huggingface.co/datasets/allegro/klej-dyk"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME
    
    def get_url(self):
        return self.HOMEPAGE_URL
