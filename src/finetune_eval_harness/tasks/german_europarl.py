
"""
German Europarl NER data (PPL only)
http://www.nlpado.de/~sebastian/pub/papers/konvens10_faruqui.pdf

We present a freely available optimized Named Entity Recognizer (NER) for German.
It alleviates the small size of available NER training corpora for German with
distributional generalization features trained on large unlabelled corpora.
e vary the size and source of the generalization corpus and find improvements
of 6% F1-score (in-domain) and 9% (out-of-domain) over simple supervised training.

Dataset: https://nlpado.de/~sebastian/software/ner_german.shtml

NOTE: This dataset is used as language modeling tasks (perplexity) and NOT named entity recogniton.

"""

from .classification import Classification
from .ner import NamedEntityRecognition


_DESCRIPTION = """
NER task for german subsplit of the Europarl dataset
"""


_CITATION = """
@InProceedings{faruqui10:_training
  author =       {Manaal Faruqui and Sebastian Pad\'{o}},
  title =        {Training and Evaluating a German Named Entity Recognizer
                  with Semantic Generalization},
  booktitle = {Proceedings of KONVENS 2010},
  year =         2010,
  address =      {Saarbr\"ucken, Germany}}
}
"""



class GermanEuroParl(NamedEntityRecognition):

    """
    Class for German Europarl Task
    """
    
    DATASET_ID = "akash418/german_europarl"
    TASK_NAME = "german_europarl"
    HOMEPAGE_URL = "https://huggingface.co/datasets/akash418/german_europarl"
    LANGUAGE = "de"

    def get_dataset_id(self):
        """
        return HF dataset id
        """
        return self.DATASET_ID

    def get_task_name(self):
        """
        return task name
        """
        return self.TASK_NAME

    def get_url(self):
        """
        return url
        """
        return self.HOMEPAGE_URL
