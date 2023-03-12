from tasks.qa import QuestionAnswering


class GermanQuad(QuestionAnswering):

    """
    Class for German Quad Task
    """

    DATASET_ID = "deepset/germanquad"  # HF datasets ID
    TASK_NAME = "german_quad"

    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME
