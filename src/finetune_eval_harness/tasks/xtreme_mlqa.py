"""

XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization

MLQA

"""


from .base.ner_task import NamedEntityRecognitionTask


_DESCRIPTION = """

"""


_CITATION = """

@article{hu2020xtreme,
      author    = {Junjie Hu and Sebastian Ruder and Aditya Siddhant and Graham Neubig and Orhan Firat and Melvin Johnson},
      title     = {XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization},
      journal   = {CoRR},
      volume    = {abs/2003.11080},
      year      = {2020},
      archivePrefix = {arXiv},
      eprint    = {2003.11080}
}

"""


class XtremeMLQABase(NamedEntityRecognitionTask):
    DATASET_ID = "xtreme"
    HOMEPAGE_URL = "https://huggingface.co/datasets/xtreme"
    LABEL_NAME = "ner_tags"


class XtremePanxEN(XtremePanxBase):
    DATASET_SPLIT = "PAN-X.en"
    TASK_NAME = "xtreme_panx_ner_en"
    LANGUAGE = "en"


class XtremePanxDE(XtremePanxBase):
    DATASET_SPLIT = "PAN-X.de"
    TASK_NAME = "xtreme_panx_ner_de"
    LANGUAGE = "de"


class XtremePanxFR(XtremePanxBase):
    DATASET_SPLIT = "PAN-X.fr"
    TASK_NAME = "xtreme_panx_ner_fr"
    LANGUAGE = "fr"


class XtremePanxES(XtremePanxBase):
    DATASET_SPLIT = "PAN-X.es"
    TASK_NAME = "xtreme_panx_ner_es"
    LANGUAGE = "es"


class XtremePanxIT(XtremePanxBase):
    DATASET_SPLIT = "PAN-X.it"
    TASK_NAME = "xtreme_panx_ner_it"
    LANGUAGE = "it"
