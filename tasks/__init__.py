from typing import List, Union
from . import (
    germeval2018,
    germeval2017,
    gnad10,
    german_ner,
    german_europarl,
    german_quad,
)

# mapping task to class objects
TASK_REGISTRY = {
    "germeval2018": germeval2018.GermEval2018,
    "germeval2017": germeval2017.GermEval2017,
    "gnad10": gnad10.Gnad10,
    "german_ner_legal": german_ner.GermanNerLegal,
    "german_europarl": german_europarl.GermanEuroParl,
    "german_quad": german_quad.GermanQuad,
}

# mapping task to type
TASK_TYPE_REGISTRY = {
    "germeval2018": "classification",
    "germeval2017": "classification",
    "gnad10": "classification",
    "german_ner_legal": "ner",
    "german_europarl": "ner",
    "german_quad": "qa",
}

ALL_TASKS = sorted(list(TASK_REGISTRY))
ALL_TASK_TYPES = sorted(list(TASK_TYPE_REGISTRY))

'''
# returning task class
def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        print("Available tasks:")
        print(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}") from exc

'''

# return string names of all the tasks for reference
def get_all_tasks():
    all_task_str = []
    for key in TASK_REGISTRY:
        all_task_str.append(key)

    return all_task_str

