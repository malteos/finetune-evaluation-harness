import datasets 
import numpy as np 
#from . import TASK_REGISTRY, TASK_TYPE_REGISTRY, get_task


_CITATION = """
@inproceedings{vamvas2020germeval,
    author    = "Wiegand, Michael, and Siegel, Melanie and Ruppenhofer, Josef",
    title     = "Overview of the GermEval 2018 Shared Task on the Identification of Offensive Language",
    booktitle = "Proceedings of the GermEval 2018 Workshop  14th Conference on Natural Language Processing (KONSENS)",
    address   = "Vienna, SAustria",
    year      = "2018",
    month     = "sep",
    url       = "https://epub.oeaw.ac.at/0xc1aa5576_0x003a10d2.pdf"
}"""


class GermEval2018():

    DATASET_ID = "philschmid/germeval18"    # HF datasets ID
    VERSION = "0"
    EPOCHS = "1"
    TRAIN_BATCH_SIZE = "16"
    MAX_SEQUENCE_LENGTH = "512"

    def __init__(self):
        super().__init__()
        
    def get_task_type(self):
        return "classification"
    
    def has_training_docs(self):
        return True
    
    def has_validation_docs(self):
        return True
    
    # method for creating training paramaeters 
    def create_train_params(
        self, 
        model_name, 
        base_checkpoint_dir, 
        logging_dir, 
        freeze_layers,
        epochs,
        per_device_batch_train_size,
        save_steps
    ):

        all_param_list = []
        all_param_list.append("--model_name_or_path")
        all_param_list.append(model_name)
        all_param_list.append("--dataset_name")
        all_param_list.append(self.DATASET_ID)
        all_param_list.append("--do_train")
        all_param_list.append("--do_eval")
        all_param_list.append("--results_log_path")
        all_param_list.append(logging_dir)
        all_param_list.append("--output_dir")
        all_param_list.append(base_checkpoint_dir)
        all_param_list.append("--overwrite_output_dir")
        all_param_list.append("True")
        all_param_list.append("--num_train_epochs")
        all_param_list.append(epochs)
        all_param_list.append("--per_device_train_batch_size")
        all_param_list.append(per_device_batch_train_size)
        all_param_list.append("--freeze_layers")
        all_param_list.append(str(freeze_layers))
        all_param_list.append("--save_steps")
        all_param_list.append(save_steps)


        print(all_param_list)
        return all_param_list