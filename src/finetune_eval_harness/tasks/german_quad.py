"""
GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval
https://arxiv.org/abs/2104.12741

In order to raise the bar for non-English QA, we are releasing a high-quality, human-labeled German QA dataset consisting of 13 722 questions, incl. a three-way annotated test set. The creation of GermanQuAD is inspired by insights from existing datasets as well as our labeling experience from several industry projects. We combine the strengths of SQuAD, such as high out-of-domain performance, with self-sufficient questions that contain all relevant information for open-domain QA as in the NaturalQuestions dataset. Our training and test datasets do not overlap like other popular datasets and include complex questions that cannot be answered with a single entity or only a few words.

Homepage: https://www.deepset.ai/germanquad
"""

from .qa import QuestionAnswering

_DESCRIPTION = """
German version of QUAD dataset inspired by SQuAD
"""


_CITATION = """

@misc{möller2021germanquad,
      title={GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval}, 
      author={Timo Möller and Julian Risch and Malte Pietsch},
      year={2021},
      eprint={2104.12741},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

"""


class GermanQuad(QuestionAnswering):

    """
    Class for German Quad Task
    """

    DATASET_ID = "deepset/germanquad"  # HF datasets ID
    TASK_NAME = "german_quad"
    HOMEPAGE_URL = "https://huggingface.co/datasets/deepset/germanquad"
    LANGUAGE = "de"
