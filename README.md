# Finetune-Evaluation-Harness

## Overview
This project is a unified framework for evaluation of various LLMs on a large number of different evaluation tasks. Some of the features of this framework:

- Different types of tasks supported: Classification, NER tagging, Question-Answering
- Support for parameter efficient tuning (PEFT)
- Running mutliple tasks altogether


## Basic Usage

To evaluate a model (eg GERMAN-BERT) on task, please use something like this:

```
python main.py --model_name_or_path bert-base-german-cased \
--task_list germeval2018 \
--results_logging_dir /sample/directory \
--output_dir /sample/directory
```

This framework is build on top of Huggingface, hence all the keyword arguments used in regular HF transformers library work here as well: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py.


## Some Mandatory Arguments

```
--model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from huggingface.co/models (default: None)

--task_list TASK_LIST [TASK_LIST ...]
                        List of tasks passed in order. (default: None)

--results_logging_dir RESULTS_LOGGING_DIR
                        Where do you want to store the pretrained models downloaded from huggingface.co (default: None)

--output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written. (default: None)



```

If you fail to understand what any of the paramater does, --help is your friend.

## List of Supported Tasks

- GNAD10 (de) https://huggingface.co/datasets/gnad10
- GermEval 2017 (de) https://huggingface.co/datasets/malteos/germeval2017
- German Europarl (de) https://huggingface.co/datasets/akash418/german_europarl
- GermEval 2018 (de) https://huggingface.co/datasets/philschmid/germeval18
- German XQUAD (de) https://huggingface.co/datasets/deepset/germanquad

