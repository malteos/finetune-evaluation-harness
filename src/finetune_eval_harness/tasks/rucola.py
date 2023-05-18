# from tasks.classification import Classification
from .classification import Classification


_DESCRIPTION = """

Russian Corpus of Linguistic Acceptability (RuCoLA) is a novel benchmark of 13.4k sentences labeled as acceptable or not. RuCoLA combines in-domain sentences manually collected from linguistic literature and out-of-domain sentences produced by nine machine translation and paraphrase generation models. 

"""


_CITATION = """

@inproceedings{mikhailov-etal-2022-rucola,
    title = "{R}u{C}o{LA}: {R}ussian Corpus of Linguistic Acceptability",
    author = "Mikhailov, Vladislav  and
      Shamardina, Tatiana  and
      Ryabinin, Max  and
      Pestova, Alena  and
      Smurov, Ivan  and
      Artemova, Ekaterina",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.348",
    pages = "5207--5227",
    abstract = "Linguistic acceptability (LA) attracts the attention of the research community due to its many uses, such as testing the grammatical knowledge of language models and filtering implausible texts with acceptability classifiers.However, the application scope of LA in languages other than English is limited due to the lack of high-quality resources.To this end, we introduce the Russian Corpus of Linguistic Acceptability (RuCoLA), built from the ground up under the well-established binary LA approach. RuCoLA consists of 9.8k in-domain sentences from linguistic publications and 3.6k out-of-domain sentences produced by generative models. The out-of-domain set is created to facilitate the practical use of acceptability for improving language generation.Our paper describes the data collection protocol and presents a fine-grained analysis of acceptability classification experiments with a range of baseline approaches.In particular, we demonstrate that the most widely used language models still fall behind humans by a large margin, especially when detecting morphological and semantic errors. We release RuCoLA, the code of experiments, and a public leaderboard to assess the linguistic competence of language models for Russian.",
}

"""


class Rucola(Classification):

    """
    Class for Rucola classification task
    """

    DATASET_ID = "RussianNLP/rucola"  # HF datasets ID
    TASK_NAME = "rucola"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/RussianNLP/rucola"

