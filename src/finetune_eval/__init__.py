

## please make sure that the vesion numbers are same here and setup.py

__version__ = "0.6.0.dev1"

from .hf_scripts import (
    data_trainining_args,
    hgf_fine_tune_class,
    hgf_fine_tune_ner,
    hgf_fine_tune_qa,
    initial_arguments,
    model_args,
    trainer_qa,
    utility_functions,
    utils_qa,
)

from .tasks import (
    classification,
    german_europarl,
    german_ner,
    germeval2017,
    germeval2018,
    gnad10,
    ner,
    qa,
    german_quad,
)

import finetune_eval.process_args as process_args
import finetune_eval.main as main
