#from tasks.ner import NamedEntityRecognition
from .ner import NamedEntityRecognition


DESCRIPTION = """

emi-manually defined in this case as: Bootstrapped from Swedish Gazetters then manually correcte/reviewed by two independent native speaking swedish annotators

"""


_CITATION = """


"""


class SwedishNer(NamedEntityRecognition):

    """
    Class for Swedish NER
    """
    
    DATASET_ID = "swedish_ner_corpus"  # HF datasets ID
    TASK_NAME = "swedish_ner"
    HOMEPAGE_URL = "https://huggingface.co/datasets/swedish_ner_corpus"
    LABEL_NAME = "ner_tags"

