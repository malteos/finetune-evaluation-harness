from tasks.classification import Classification



class KlejDyk(Classification):

    """
    Class for Klej Dyk Classification Task
    """


    DATASET_ID = "allegro/klej-dyk"  # HF datasets ID
    TASK_NAME = "klej_dyk"
    LABEL_NAME = "target"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/allegro/klej-dyk"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
