from tasks.classification import Classification



class Rucola(Classification):

    """
    Class for Rucola classification task
    """


    DATASET_ID = "RussianNLP/rucola"  # HF datasets ID
    TASK_NAME = "rucola"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/RussianNLP/rucola"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
