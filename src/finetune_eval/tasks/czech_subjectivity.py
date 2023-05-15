from tasks.classification import Classification



class CzechSubjectivity(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "pauli31/czech-subjectivity-dataset"  # HF datasets ID
    TASK_NAME = "czech_subjectivity"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/pauli31/czech-subjectivity-dataset"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL
    
