from tasks.classification import Classification


class Gnad10(Classification):

    DATASET_ID = "gnad10"  # HF datasets ID
    TASKNAME = "gnad10"
    LABEL_NAME = "label"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_label_name(self):
        return self.LABEL_NAME
