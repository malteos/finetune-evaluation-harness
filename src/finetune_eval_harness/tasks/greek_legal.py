"""

Greek_Legal_Code (GLC) is a dataset consisting of approx. 47k legal resources from Greek legislation. 
The origin of GLC is “Permanent Greek Legislation Code - Raptarchis”, a collection of Greek legislative 
documents classified into multi-level (from broader to more specialized) categories.


GLC consists of 47 legislative volumes and each volume corresponds to a main thematic topic. 
Each volume is divided into thematic sub categories which are called chapters and subsequently, 
each chapter breaks down to subjects which contain the legal resources. The total number of 
chapters is 389 while the total number of subjects is 2285, creating an interlinked thematic hierarchy. 
So, for the upper thematic level (volume) GLC has 47 classes. For the next thematic level (chapter) 
GLC offers 389 classes and for the inner and last thematic level (subject), GLC has 2285 classes.

GLC classes are divided into three categories for each thematic level: frequent classes, 
which occur in more than 10 training documents and can be found in all three subsets 
(training, development and test); few-shot classes which appear in 1 to 10 training documents 
and also appear in the documents of the development and test sets, and zero-shot classes 
which appear in the development and/or test, but not in the training documents.

"""


from .classification import Classification


_DESCRIPTION = """
Dataset consisting of greek legal resources from Greek legislation
"""


_CITATION = """

@inproceedings{papaloukas-etal-2021-glc,
    title = "Multi-granular Legal Topic Classification on Greek Legislation",
    author = "Papaloukas, Christos and Chalkidis, Ilias and Athinaios, Konstantinos and Pantazi, Despina-Athanasia and Koubarakis, Manolis",
    booktitle = "Proceedings of the 3rd Natural Legal Language Processing (NLLP) Workshop",
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "",
    url = "https://arxiv.org/abs/2109.15298",
    doi = "",
    pages = ""
}

"""



class GreekLegal(Classification):


    DATASET_ID = "greek_legal_code"  
    TASK_NAME = "greek_legal"
    LABEL_NAME = "label"  
    HOMEPAGE_URL = "https://huggingface.co/datasets/greek_legal_code"
    LANGUAGE = "el"
    
