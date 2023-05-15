from tasks.classification import Classification



class FinishSentiment(Classification):

    """
    Class for Finish Sentiment Classification
    """


    DATASET_ID = "sepidmnorozy/Finnish_sentiment"  # HF datasets ID
    TASK_NAME = "finish_sentiment"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Finnish_sentiment"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
