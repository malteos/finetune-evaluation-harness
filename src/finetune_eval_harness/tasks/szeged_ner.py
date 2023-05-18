#from tasks.ner import NamedEntityRecognition
from .ner import NamedEntityRecognition


DESCRIPTION = """

The recognition and classification of proper nouns and names in plain text is of key importance in Natural Language Processing (NLP) as it has a beneficial effect on the performance of various types of applications, including Information Extraction, Machine Translation, Syntactic Parsing/Chunking, etc.

"""


_CITATION = """

@article{szarvas2006highly,
  title={A highly accurate Named Entity corpus for Hungarian},
  author={Szarvas, Gy{\"o}rgy and Farkas, Rich{\'a}rd and Felf{\"o}ldi, L{\'a}szl{\'o} and Kocsor, Andr{\'a}s and Csirik, J{\'a}nos},
  journal={annotation},
  volume={2},
  pages={3--1},
  year={2006},
  publisher={Citeseer}
}

"""




class SzegedNer(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "ficsort/SzegedNER"  # HF datasets ID
    TASK_NAME = "szeged_ner"
    HOMEPAGE_URL = "https://huggingface.co/datasets/ficsort/SzegedNER"
    LABEL_NAME = "ner"

    
