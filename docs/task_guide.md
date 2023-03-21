# `Task` Guide

The `Task` class is the foundation of all natural language tasks in the `lm-evaluation-harness` (harness). It encompasses everything you’d need to perform few-shot evaluation of an autoregressive language model. Here we’ll provide a step-by-step guide on how to subclass `Task` to create your very own task/s.

## Setup

If you haven't already, go ahead and fork the main repo, clone it, create a branch with the name of your task, and install the project requirements in your environment:

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

### Running Unit Tests

To run the entire test suite, use:

```sh
coverage run -m pytest
```

To generate a code coverage report for the entire test suit

```sh
coverage report -m
```