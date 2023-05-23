"""

CoNLL-NERC is the Spanish dataset of the CoNLL-2002 Shared Task (Tjong Kim Sang, 2002). 
The dataset is annotated with four types of named entities --persons, locations, organizations, 
and other miscellaneous entities-- formatted in the standard Beginning-Inside-Outside (BIO) format. 
The corpus consists of 8,324 train sentences with 19,400 named entities, 1,916 development sentences 
with 4,568 named entities, and 1,518 test sentences with 3,644 named entities.


"""



from .classification import Classification
from .ner import NamedEntityRecognition

_DESCRIPTION = """

CoNLL-NERC is the Spanish dataset of the CoNLL-2002 Shared Task (Tjong Kim Sang, 2002). The dataset is annotated with four types of named entities --persons, locations, organizations, and other miscellaneous entities-- formatted in the standard Beginning-Inside-Outside (BIO) format. 

"""


_CITATION = """

@article{sang2003introduction,
  title={Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition},
  author={Sang, Erik F and De Meulder, Fien},
  journal={arXiv preprint cs/0306050},
  year={2003}
}

"""


class SpanishConll(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "PlanTL-GOB-ES/CoNLL-NERC-es"  
    TASK_NAME = "spanish_conll"
    HOMEPAGE_URL = "https://huggingface.co/datasets/PlanTL-GOB-ES/CoNLL-NERC-es"
    LANGUAGE = "es"
    
