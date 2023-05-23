"""
Germeval Task 2017: Shared Task on Aspect-based Sentiment in Social Media Customer Feedback

Paper: https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2017-wojatzkietal-germeval2017-workshop.pdf

Huggingface dataset: https://huggingface.co/datasets/malteos/germeval2017

Original dataset: http://ltdata1.informatik.uni-hamburg.de/germeval2017/
"""


from .classification import Classification


_DESCRIPTION = """

Germeval Task 2017: Shared Task on Aspect-based Sentiment in Social Media Customer Feedback

"""


_CITATION = """

@inproceedings{germevaltask2017,
title = {{GermEval 2017: Shared Task on Aspect-based Sentiment in Social Media Customer Feedback}},
author = {Michael Wojatzki and Eugen Ruppert and Sarah Holschneider and Torsten Zesch and Chris Biemann},
year = {2017},
booktitle = {Proceedings of the GermEval 2017 - Shared Task on Aspect-based Sentiment in Social Media Customer Feedback},
address={Berlin, Germany},
pages={1--12}
}

"""


class GermEval2017(Classification):

    

    DATASET_ID = "akash418/germeval_2017"  
    TASK_NAME = "germeval2017"
    LABEL_NAME = "relevance"  
    HOMEPAGE_URL = "https://huggingface.co/datasets/malteos/germeval2017"
    LANGUAGE = "de"

