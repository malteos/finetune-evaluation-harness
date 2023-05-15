from tasks.classification import Classification


_DESCRIPTION = """

"""

class BulgarianSentiment(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "sepidmnorozy/Bulgarian_sentiment"  # HF datasets ID
    TASK_NAME = "bulgarian_sentiment"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/sepidmnorozy/Bulgarian_sentiment"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
