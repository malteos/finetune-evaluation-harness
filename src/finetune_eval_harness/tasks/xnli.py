"""

XNLI is a subset of a few thousand examples from MNLI which has been translated 
into a 14 different languages (some low-ish resource). As with MNLI, the goal is to predict textual entailment 
(does sentence A imply/contradict/neither sentence B) and is a classification task (given two sentences, predict one of three labels).


"""



_CITATION = """

@InProceedings{conneau2018xnli,
  author = {Conneau, Alexis
                 and Rinott, Ruty
                 and Lample, Guillaume
                 and Williams, Adina
                 and Bowman, Samuel R.
                 and Schwenk, Holger
                 and Stoyanov, Veselin},
  title = {XNLI: Evaluating Cross-lingual Sentence Representations},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  location = {Brussels, Belgium},
}



"""


from .classification import Classification


class Xnli(Classification):
    DATASET_ID = "xnli"           
    TASK_NAME = "xnli"
    LABEL_NAME = "label"            
    HOMEPAGE_URL = "https://huggingface.co/datasets/xnli"



class XnliDe(Xnli):
    DATASET_SPLIT = "de"
    TASK_NAME = "xnli_de"


class XnliEs(Xnli):
    DATASET_SPLIT = "es"
    TASK_NAME = "xnli_es"


class XnliEn(Xnli):
    DATASET_SPLIT = "em"
    TASK_NAME = "xnli_en"