from tasks.classification import Classification


class Gnad10(Classification):

    """
    Class for GNAD10 Classification Task
    """
    
    DATASET_ID = "gnad10"  # HF datasets ID
    TASKNAME = "gnad10"
    LABEL_NAME = "label"
    HOMEPAGE_URL = "https://huggingface.co/datasets/gnad10"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_label_name(self):
        return self.LABEL_NAME

    def get_task_name(self):
        return self.TASKNAME

    def get_url(self):
        return self.HOMEPAGE_URL