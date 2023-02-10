import datasets 
import numpy as np 
#from . import TASK_REGISTRY, TASK_TYPE_REGISTRY, get_task



class GermanEuroParl():

    DATASET_ID = "akash418/german_europarl"    # HF datasets ID
    VERSION = "0"
    EPOCHS = "1"
    TRAIN_BATCH_SIZE = "16"
    MAX_SEQUENCE_LENGTH = "512"

    def __init__(self):
        super().__init__()
        
    def get_task_type(self):
        return "ner"
    
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
        per_device_train_batch_size,
        save_steps,
        peft_choice,
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
        all_param_list.append(per_device_train_batch_size)
        all_param_list.append("--freeze_layers")
        all_param_list.append(str(freeze_layers))
        all_param_list.append("--save_steps")
        all_param_list.append(save_steps)
        all_param_list.append("--peft_choice"),
        all_param_list.append(peft_choice)
        all_param_list.append("--is_task_ner")
        all_param_list.append("True")

        print(all_param_list)
        return all_param_list