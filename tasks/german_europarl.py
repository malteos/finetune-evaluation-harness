from tasks.ner import NamedEntityRecognition


class GermanEuroParl(NamedEntityRecognition):

    DATASET_ID = "akash418/german_europarl"
    TASK_NAME = "german_europarl"

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
