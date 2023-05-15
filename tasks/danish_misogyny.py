from tasks.classification import Classification



class DanishMisogyny(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "strombergnlp/bajer_danish_misogyny"  # HF datasets ID
    TASK_NAME = "danish_misogyny"
    LABEL_NAME = "subtask_A"            # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/strombergnlp/bajer_danish_misogyny"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
