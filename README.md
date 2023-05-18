# Finetune-Evaluation-Harness

![Build Status](https://github.com/malteos/finetune-evaluation-harness/actions/workflows/coverage_eval.yml/badge.svg)
![Build Status](https://github.com/malteos/finetune-evaluation-harness/actions/workflows/pull_request.yml/badge.svg)


## Overview
This project is a unified framework for evaluation of various LLMs on a large number of different evaluation tasks. Some of the features of this framework:

- Different types of tasks supported: Classification, NER tagging, Question-Answering
- Support for parameter efficient tuning (PEFT)
- Running mutliple tasks altogether


## Getting Started
To evaluate a model (eg GERMAN-BERT) on task, please use something like this:

```python
import finetune_eval_harness


finetune-eval-harness --model_name_or_path bert-base-german-cased \
--task_list germeval2018 \
--results_logging_dir /sample/directory/results \
--output_dir /sample/directory/results


````

Please refer to the latest package details here: https://pypi.org/project/finetune-eval-harness/

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

## Some of the Tasks

- GNAD10 (de) https://huggingface.co/datasets/gnad10
- GermEval 2017 (de) https://huggingface.co/datasets/malteos/germeval2017
- German Europarl (de) https://huggingface.co/datasets/akash418/german_europarl
- GermEval 2018 (de) https://huggingface.co/datasets/philschmid/germeval18
- German XQUAD (de) https://huggingface.co/datasets/deepset/germanquad


For a detailed list of tasks, please use

```python
finetune_eval_harness.get_all_tasks()


['germeval2018', 'germeval2017', 'gnad10', 'german_ner_legal', 'german_europarl', 'german_quad', 'spanish_quad', 'wiki_cat_es', 'spanish_conll', 'flue', 'spanish_ehealth', 'szeged_ner', 'polish_dyk', 'mapa', 'eur_lux', 'ehealth_kd', 'rucola', 'klej_dyk', 'croatian_sentiment', 'finish_sentiment', 'swedish_ner', 'greek_legal', 'bulgarian_sentiment', 'czech_subjectivity', 'danish_misogyny', 'slovak_sentiment', 'maltese_sentiment', 'dutch_social']

````


## Implementing New Tasks

To implement a new task in eval harness, see [this guide](./docs/task_guide.md).


## Evaluating the Coverage of the Current Code
Please go to Github Actions sections of this repository and start the build named "Evaluate", this would check if the coverage on existing code is more than 80%. The build
status is also visible on the main repo page.

## Guidelines On Running Tasks
- In some instances for specific tasks, please make sure to specify the exact dataset config depending on your needs
- If text sequence processing fails for some tasks such as classification, please try with setting --use_fast_tokenizer as False
- Please make sure that the dataset (i.e task) url is publically visible on huggingface datasets

