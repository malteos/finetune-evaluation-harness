import sys
import evaluate
from transformers import (
    DataCollatorForTokenClassification,
    PretrainedConfig,
    set_seed,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version
from datasets import ClassLabel
from hf_scripts.utility_functions import *
import hf_scripts

def run_task_evaluation(model_args, data_args, training_args, init_args):

    #model_args, data_args, training_args = hf_scripts.utility_functions.parse_hf_arguments(args)
    (training_args, data_args) = hf_scripts.utility_functions.prepend_data_args(training_args, data_args, init_args)
    send_example_telemetry("run_ner", model_args, data_args)

    logger = hf_scripts.utility_functions.prepare_logger(training_args)
    last_checkpoint = hf_scripts.utility_functions.detect_last_checkpoint(logger, training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = hf_scripts.utility_functions.load_raw_dataset_ner(data_args, model_args)

    # raw_datasets = raw_datasets.map(lambda x: {"ner_tags": [int(i) for i in x["ner_tags"].split(",")]})
    # raw_datasets = raw_datasets.map(lambda x: {"pos_tags": [int(i) for i in x["pos_tags"].split(",")]})

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    # elif f"{data_args.task_name}_tags" in column_names:
    #    label_column_name = f"{data_args.task_name}_tags"
    elif data_args.is_task_ner:
        label_column_name = "ner_tags"
    else:
        label_column_name = column_names[1]

    # label_list = get_label_list(labels)
    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.

    label_list, label_to_id, feature_file_exists, labels_are_int = hf_scripts.utility_functions.generate_label_list(
        raw_datasets, features, ClassLabel, label_column_name
    )

    num_labels = len(label_list)
    logger.info(f"Number of distinct labels {num_labels}")

    config = hf_scripts.utility_functions.load_config(
        model_args.model_name_or_path,
        num_labels,
        "ner",
        model_args.cache_dir,
        model_args.model_revision,
        model_args.use_auth_token,
        "ner",
    )
    tokenizer = hf_scripts.utility_functions.load_tokenizer(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        model_args.cache_dir,
        model_args.use_fast_tokenizer,
        model_args.model_revision,
        model_args.use_auth_token,
        None,
        True if config.model_type in {"bloom", "gpt2", "roberta"} else False,
    )

    model = hf_scripts.utility_functions.load_model(
        model_args.model_name_or_path,
        bool(".ckpt" in model_args.model_name_or_path),
        config,
        model_args.cache_dir,
        model_args.model_revision,
        model_args.use_auth_token,
        "ner",
    )

    hf_scripts.utility_functions.print_trainable_parameters(model)

    if data_args.peft_choice in hf_scripts.utility_functions.peft_choice_list:
        model = hf_scripts.utility_functions.load_model_peft(model, data_args, "TOKEN_CLS")

    model = hf_scripts.utility_functions.freeze_layers(model_args, model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))

    hf_scripts.utility_functions.check_tokenizer_instance(tokenizer)

    '''
    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {
                    i: int(model.config.label2id[l]) for i, l in enumerate(label_list)
                }
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels:"
                f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
            )
    '''

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    b_to_i_label = hf_scripts.utility_functions.generate_b_to_i_label(feature_file_exists, label_list)

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    fn_kwargs = {
        "tokenizer": tokenizer,
        "text_column_name": text_column_name,
        "padding": padding,
        "data_args": data_args,
        "label_column_name": label_column_name,
        "label_to_id": label_to_id,
        "b_to_i_label": b_to_i_label,
    }

    train_dataset, eval_dataset, predict_dataset = hf_scripts.utility_functions.map_train_validation_predict_ds_ner(
        training_args, data_args, raw_datasets, fn_kwargs
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Metrics
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        return hf_scripts.utility_functions.get_true_predictions_labels(label_list, predictions, labels, metric, data_args)

    metrics_eval = hf_scripts.utility_functions.train_eval_prediction(
        "token-classification",
        model,
        training_args,
        data_args,
        model_args,
        train_dataset,
        eval_dataset,
        None,
        data_collator,
        tokenizer,
        None,
        compute_metrics,
        last_checkpoint,
        None,
        predict_dataset,
        None,
        label_list,
        False,
    )

    #trainer = hf_scripts.utility_functions.set_hub_arguments(
    #    trainer, model_args, data_args, training_args, "token-classification"
    #)

    logger.info(f"Training Metrics {metrics_eval}")
    return metrics_eval

def main():
    run_task_evaluation()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    # main()
    main(sys.argv[1:])
