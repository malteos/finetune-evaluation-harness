
# TODO import statement for the new tasktype (replace it)
from tasks.new_task_type import NewTaskType


# TODO: Add the BibTeX citation for the task, if it exists.
_CITATION = """
"""

# TODO: Replace `NewTask` and `NewTaskType` with the name of your Task and Task Type respectively.
class NewTask(NewTaskType):

    """
    Template Class for defining New Task
    """

    # TODO: Add the `DATASET_ID` string. 
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_ID = ""

    # TODO: Add the `TASK_NAME` string. This will be the name of the `Task`
    TASK_NAME = ""
    
    # TODO: Add the `LABEL_NAME` string. If it exists
    LABEL_NAME = ""
    
    # TODO: Add the `HOMEPAGE_URL` string. URL Linking to HF dataset
    HOMEPAGE_URL = ""

    # TODO: Add the more getter methods if needed


    def get_dataset_id(self):
        return self.DATASET_ID

    def get_task_name(self):
        return self.TASK_NAME

    def get_label_name(self):
        return self.LABEL_NAME

    def get_url(self):
        return self.HOMEPAGE_URL

