"""
The x-stance dataset contains more than 150 political questions, and 67k comments written by candidates on those questions.

It can be used to train and evaluate stance detection systems. The comments are partly German, partly French and Italian. 
The questions are available in all the three languages plus English. The data have been extracted from the Swiss voting advice platform Smartvote.
"""


_CITATION = """
@inproceedings{vamvas2020xstance,
    author    = "Vamvas, Jannis and Sennrich, Rico",
    title     = "{X-Stance}: A Multilingual Multi-Target Dataset for Stance Detection",
    booktitle = "Proceedings of the 5th Swiss Text Analytics Conference (SwissText) \& 16th Conference on Natural Language Processing (KONVENS)",
    address   = "Zurich, Switzerland",
    year      = "2020",
    month     = "jun",
    url       = "http://ceur-ws.org/Vol-2624/paper9.pdf"
}
"""


from .base.classification_task import ClassificationTask


class XStanceBase(ClassificationTask):
    DATASET_ID = "x_stance"
    TASK_NAME = "xstance"
    LABEL_NAME = "label"
    HOMEPAGE_URL = "https://github.com/ZurichNLP/xstance"

    LANGUAGE = None

    def get_text_column_names(self):
        return "question", "comment"

    def filter_dataset(self, example):
        if example["language"] == self.LANGUAGE:
            return True
        else:
            return False


class XStanceDE(XStanceBase):
    LANGUAGE = "de"


class XStanceFR(XStanceBase):
    LANGUAGE = "fr"


class XStanceIT(XStanceBase):
    LANGUAGE = "it"
