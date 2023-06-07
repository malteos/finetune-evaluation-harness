from hf_scripts import hgf_fine_tune_ner
from tasks.base import BaseTask


class NamedEntityRecognitionTask(BaseTask):
    def get_task_type(self) -> str:
        return "ner"

    def run_task_evaluation(self):
        return hgf_fine_tune_ner.run_task_evaluation(
            self.model_args, self.data_args, self.training_args, self.init_args
        )
