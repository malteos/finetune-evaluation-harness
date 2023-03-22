from tasks.ner import NamedEntityRecognition


class GermanEuroParl(NamedEntityRecognition):

    """
    Class for German Europarl Task
    """

    DATASET_ID = "akash418/german_europarl"
    TASK_NAME = "german_europarl"
    HOMEPAGE_URL = "https://huggingface.co/datasets/akash418/german_europarl"

    def get_dataset_id(self):
        """
        return HF dataset id
        """
        return self.DATASET_ID

    def get_task_name(self):
        """
        return task name
        """
        return self.TASK_NAME

    def get_url(self):
        """
        return url
        """
        return self.HOMEPAGE_URL
