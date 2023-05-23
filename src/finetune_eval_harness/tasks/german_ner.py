
"""
A dataset of Legal Documents from German federal court decisions for Named Entity Recognition.
https://arxiv.org/abs/2003.13016v1

The dataset is human-annotated with 19 fine-grained entity classes.
The dataset consists of approx. 67,000 sentences and contains 54,000 annotated entities.
NER tags use the BIO tagging scheme.

Dataset: https://huggingface.co/datasets/elenanereiss/german-ler

"""



_CITATION = """
@misc{german-ler,
  doi = {10.48550/ARXIV.2003.13016},
  url = {https://arxiv.org/abs/2003.13016},
  author = {Leitner, Elena and Rehm, Georg and Moreno-Schneider, Julián},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Dataset of German Legal Documents for Named Entity Recognition},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""



from .ner import NamedEntityRecognition

class GermanNerLegal(NamedEntityRecognition):

    """
    Class for German NER Legal Task
    """
    
    DATASET_ID = "elenanereiss/german-ler"  
    TASK_NAME = "german_ner"
    HOMEPAGE_URL = "https://huggingface.co/datasets/elenanereiss/german-ler"

    
