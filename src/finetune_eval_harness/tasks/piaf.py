"""

Piaf is a reading comprehension dataset. This version, published in February 2020, contains 3835 questions on French Wikipedia.


"""


from .base.qa_task import QuestionAnsweringTask


class Piaf(QuestionAnsweringTask):
    DATASET_ID = "etalab-ia/piaf"
    TASK_NAME = "piaf"
    HOMEPAGE_URL = "https://huggingface.co/datasets/etalab-ia/piaf"
    LANGUAGE = "fr"
