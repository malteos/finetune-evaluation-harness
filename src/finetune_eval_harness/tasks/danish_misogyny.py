
"""

This is a high-quality dataset of annotated posts sampled from social media posts and annotated for misogyny. Danish language.

Online misogyny, a category of online abusive language, has serious and harmful social consequences. 
Automatic detection of misogynistic language online, while imperative, poses complicated challenges to both data gathering, data annotation, and bias mitigation, as this type of data is linguistically complex and diverse.

See the accompanying ACL paper Annotating Online Misogyny for full details.


"""

from .classification import Classification


_DESCRIPTION = """
Classification of online misogyny in Danish language with expert labels 
"""


_CITATION = """
@inproceedings{zeinert-etal-2021-annotating,
    title = "Annotating Online Misogyny",
    author = "Zeinert, Philine  and
      Inie, Nanna  and
      Derczynski, Leon",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.247",
    doi = "10.18653/v1/2021.acl-long.247",
    pages = "3181--3197",
}
"""


class DanishMisogyny(Classification):



    DATASET_ID = "strombergnlp/bajer_danish_misogyny"  
    TASK_NAME = "danish_misogyny"
    LABEL_NAME = "subtask_A"                           
    HOMEPAGE_URL = "https://huggingface.co/datasets/strombergnlp/bajer_danish_misogyny"
    LANGUAGE = "da"
    
