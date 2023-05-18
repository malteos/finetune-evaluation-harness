#from tasks.qa import QuestionAnswering
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

