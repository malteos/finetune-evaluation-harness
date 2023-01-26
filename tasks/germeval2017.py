import datasets 
import numpy as np 
#from . import TASK_REGISTRY, TASK_TYPE_REGISTRY, get_task

_CITATION = """
@inproceedings{germevaltask2017,
title = {{GermEval 2017: Shared Task on Aspect-based Sentiment in Social Media Customer Feedback}},
author = {Michael Wojatzki and Eugen Ruppert and Sarah Holschneider and Torsten Zesch and Chris Biemann},
year = {2017},
booktitle = {Proceedings of the GermEval 2017 - Shared Task on Aspect-based Sentiment in Social Media Customer Feedback},
address={Berlin, Germany},
pages={1--12}
}
"""


class GermEval2017():

    DATASET_ID = "akash418/germeval_2017"    # HF datasets ID
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
        per_device_train_batch_size,
        save_steps,
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
        all_param_list.append("--use_fast_tokenizer")
        all_param_list.append("False")
        all_param_list.append("--save_steps")
        all_param_list.append(save_steps)

        print(all_param_list)
        return all_param_list