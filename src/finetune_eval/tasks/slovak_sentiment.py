from tasks.classification import Classification



class SlovakSentiment(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "sepidmnorozy/Slovak_sentiment"  # HF datasets ID
    TASK_NAME = "slovak_sentiment"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Slovak_sentiment"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
