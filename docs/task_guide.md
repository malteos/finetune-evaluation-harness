# `Task` Guide

The `Task` class is the foundation of all natural language tasks in the `lm-evaluation-harness` (harness). It encompasses everything you’d need to perform few-shot evaluation of an autoregressive language model. Here we’ll provide a step-by-step guide on how to subclass `Task` to create your very own task/s.

## Setup

If you haven't already, go ahead and fork the main repo, clone it, create a branch with the name of your task, and install the project requirements in your environment:

```sh
# After forking...
git clone https://github.com/<YOUR-USERNAME>/finetune-evaluation-harness-evaluation-harness.git
cd finetune-evaluation-harness
git checkout -b <task-name>
pip install -r requirements.txt
```


## Handling Data
All data downloading and management is handled through the HuggingFace (**HF**) [`datasets`](https://github.com/huggingface/datasets) API. So, the first thing you should do is check to see if your task's dataset is already provided in their catalog [here](https://huggingface.co/datasets). If it's not in there, please consider adding it to their Hub to make it accessible to a wider user base by following their [new dataset guide](https://github.com/huggingface/datasets/blob/master/ADD_NEW_DATASET.md)
.

## Creating Your New Task Type File

First check if the task type on which you want to work on is different from existing ones in /tasks folder. If no and you want to just create a new task, head on the
next section of this documentation

From the `finetune-evaluation-harness` project root, copy over the `new_task_type.py` template to `finetune-evaluation-harness/tasks`.

```sh
cp templates/new_task_type.py finetune-evaluation-harness/<task-type-name>.py
```

Follow the template file to just define abstract method depending on the needs of your task type. For refrence, you can go through existing task types `tasks/classification.py` or
`tasks/ner.py`

## Creating Your New Task
Assuming you have defined a new task type or want to expand a new task for an existing task type refer to this section

rom the `finetune-evaluation-harness` project root, copy over the `new_task.py` template to `finetune-evaluation-harness/tasks`.

```sh
cp templates/new_task.py finetune-evaluation-harness/<task-name>.py
```

Next, follow the TODO's specified in the template file to impelment abstract methods depending on your needs


## Registering Tasks

Now's a good time to register your task to expose it for usage. All you'll need to do is import your task module in `finetune-evaluation-harness-evaluation-harness/tasks/__init__.py` and provide an entry in the `TASK_REGISTRY`  dictionary with the key as the name of your benchmark task (in the form it'll be referred to in the command line) and the value as the task class. See how it's done for other tasks in the [file](https://github.com/malteos/finetune-evaluation-harness/blob/main/tasks/__init__.py).


## Running Unit Tests

To run the entire test suite, use:

```sh
coverage run -m pytest
```

To generate a code coverage report for the entire test suit

```sh
coverage report -m
```