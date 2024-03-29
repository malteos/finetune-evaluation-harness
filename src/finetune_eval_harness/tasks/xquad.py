"""

XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset 
for evaluating cross-lingual question answering performance. 
The dataset consists of a subset of 240 paragraphs and 
1190 question-answer pairs from the development set of SQuAD v1.1 (
Rajpurkar et al., 2016) together with their professional translations 
into ten languages: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, 
Thai, Chinese, and Hindi. Consequently, the dataset is entirely parallel across 11 languages.

====> deprecated: XQUAD has no train set!

"""


from .base.qa_task import QuestionAnsweringTask


class XQuad(QuestionAnsweringTask):
    DATASET_ID = "xquad"
    TASK_NAME = "xquad"
    HOMEPAGE_URL = "https://huggingface.co/datasets/xquad"


class XQuadDe(XQuad):
    DATASET_SPLIT = "xquad.de"
    TASK_NAME = "xquad_de"
    LANGUAGE = "de"


class XQuadEs(XQuad):
    DATASET_SPLIT = "xquad.es"
    TASK_NAME = "xquad_es"
    LANGUAGE = "es"


class XQuadEn(XQuad):
    DATASET_SPLIT = "xquad.en"
    TASK_NAME = "xquad_en"
    LANGUAGE = "en"
