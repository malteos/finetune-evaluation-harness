"""
Webbnyheter 2012 from Spraakbanken, semi-manually annotated and adapted for CoreNLP Swedish NER.
Semi-manually defined in this case as: Bootstrapped from Swedish Gazetters then manually correcte/reviewed 
by two independent native speaking swedish annotators. No annotator agreement calculated.

"""


from .ner import NamedEntityRecognition


DESCRIPTION = """

semi-manually defined in this case as: Bootstrapped from Swedish Gazetters then manually correcte/reviewed 
by two independent native speaking swedish annotators

"""


_CITATION = """

"""


class SwedishNer(NamedEntityRecognition):
    
    DATASET_ID = "swedish_ner_corpus"  
    TASK_NAME = "swedish_ner"
    HOMEPAGE_URL = "https://huggingface.co/datasets/swedish_ner_corpus"
    LABEL_NAME = "ner_tags"
    LANGUAGE = "sv"
