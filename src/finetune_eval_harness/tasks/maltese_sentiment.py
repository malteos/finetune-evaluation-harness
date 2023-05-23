"""
PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification

The dataset consists of 23,659 human translated PAWS evaluation pairs and
296,406 machine translated training pairs in 6 typologically distinct languages.

Examples are adapted from  PAWS-Wiki

Git: https://github.com/google-research-datasets/paws/tree/master/pawsx
Paper: https://arxiv.org/abs/1908.11828

Prompt format (same as in mGPT):

"<s>" + sentence1 + ", right? " + mask + ", " + sentence2 + "</s>",

where mask is the string that matches the label:

Yes, No.

Example:

<s> The Tabaci River is a tributary of the River Leurda in Romania, right? No, The Leurda River is a tributary of the River Tabaci in Romania.</s>

Language specific prompts are translated word-by-word with Google Translate
and may differ from the ones used by mGPT and XGLM (they do not provide their prompts).

"""


from .classification import Classification

_DESCRIPTION = """
Maltese version of PAWS-X dataset
"""


_CITATION = """


"""


class MalteseSentiment(Classification):

    """
    Class for GermEval 2017 Classification Task
    """


    DATASET_ID = "amitness/PAWS-X-maltese"  
    TASK_NAME = "maltese_sentiment"
    LABEL_NAME = "label"  
    HOMEPAGE_URL = "https://huggingface.co/datasets/amitness/PAWS-X-maltese"
    LANGUAGE = "mt"
    
