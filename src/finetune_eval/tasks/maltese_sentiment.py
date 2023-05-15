from tasks.classification import Classification



class MalteseSentiment(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "amitness/PAWS-X-maltese"  # HF datasets ID
    TASK_NAME = "maltese_sentiment"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/amitness/PAWS-X-maltese"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
