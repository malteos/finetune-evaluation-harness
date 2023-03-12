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

If you fail to understand what any of the paramater does, --help is your friend.

## List of Supported Tasks

- GNAD10 (de)
- GermEval 2017 (de)
- German Europarl (de)
- GermEval 2018 (de)
- German XQUAD (de)


