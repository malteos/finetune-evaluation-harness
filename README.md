# Finetune-Evaluation-Harness

![Build Status](https://github.com/malteos/finetune-evaluation-harness/actions/workflows/coverage_eval.yml/badge.svg)
![Build Status](https://github.com/malteos/finetune-evaluation-harness/actions/workflows/pull_request.yml/badge.svg)


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
--results_logging_dir /sample/directory/results \
--output_dir /sample/directory
```

This framework is build on top of Huggingface, hence all the keyword arguments used in regular HF transformers library work here as well: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py.


## Some Important Arguments

```
--model_name_or_path MODEL_NAME_OR_PATH
    Path to pretrained model or model identifier from huggingface.co/models (default: None)

--task_list TASK_LIST [TASK_LIST ...]
    List of tasks passed in order. (default: None) eg germeval2018, germeval2017, gnad10, german_europarl

--results_logging_dir RESULTS_LOGGING_DIR
   Where do you want to save the results of the run as a json file (default: None)

--output_dir OUTPUT_DIR
	The output directory where the model predictions and checkpoints will be written. (default: None)

--num_train_epochs NUM_TRAIN_EPOCHS
    Total number of training epochs to perform. (default: 1.0)

--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
    Batch size per GPU/TPU core/CPU for training. (default: 8)

--use_fast_tokenizer [USE_FAST_TOKENIZER]
    Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. (default: True)

```

If you fail to understand what any of the paramater does, --help is your friend.

## List of Supported Tasks

- GNAD10 (de) https://huggingface.co/datasets/gnad10
- GermEval 2017 (de) https://huggingface.co/datasets/malteos/germeval2017
- German Europarl (de) https://huggingface.co/datasets/akash418/german_europarl
- GermEval 2018 (de) https://huggingface.co/datasets/philschmid/germeval18
- German XQUAD (de) https://huggingface.co/datasets/deepset/germanquad


## Implementing New Tasks

To implement a new task in eval harness, see [this guide](./docs/task_guide.md).


## Evaluating the Coverage of the Current Code
Please go to Github Actions sections of this repository and start the build named "Evaluate", this would check if the coverage on existing code is more than 80%. The build
status is also visible on the main repo page.

