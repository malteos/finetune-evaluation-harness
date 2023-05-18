#from tasks.qa import QuestionAnswering
from .qa import QuestionAnswering

DESCRIPTION = """

Automatic translation of the Stanford Question Answering Dataset (SQuAD) v2 into Spanish

"""


_CITATION = """

@article{2016arXiv160605250R,
       author = {Casimiro Pio , Carrino and  Marta R. , Costa-jussa and  Jose A. R. , Fonollosa},
        title = "{Automatic Spanish Translation of the SQuAD Dataset for Multilingual
Question Answering}",
      journal = {arXiv e-prints},
         year = 2019,
          eid = {arXiv:1912.05200v1},
        pages = {arXiv:1912.05200v1},
archivePrefix = {arXiv},
       eprint = {1912.05200v2},
}

"""




class SpanishQuad(QuestionAnswering):

    """
    Class for German Quad Task
    """

    DATASET_ID = "squad_es"  # HF datasets ID
    TASK_NAME = "es_squad"
    HOMEPAGE_URL = "https://huggingface.co/datasets/squad_es"

