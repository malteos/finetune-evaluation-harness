from tasks.classification import Classification



class GreekLegal(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "greek_legal_code"  # HF datasets ID
    TASK_NAME = "greek_legal"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/greek_legal_code"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
