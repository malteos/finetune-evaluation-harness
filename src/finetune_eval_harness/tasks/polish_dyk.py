"""

The Czy wiesz? (eng. Did you know?) the dataset consists of almost 5k question-answer pairs obtained from Czy wiesz... 
section of Polish Wikipedia. Each question is written by a Wikipedia collaborator and is answered with a link to a relevant Wikipedia article. 
In huggingface version of this dataset, they chose the negatives which have the largest token overlap with a question.


"""



from .qa import QuestionAnswering


_DESCRIPTION = """

Question Answering dataset consists of almost 5k question-answer pairs obtained from Czy wiesz, section of Polish Wikipedia

"""


_CITATION = """

@misc{11321/39,	
 title = {Pytania i odpowiedzi z serwisu wikipedyjnego "Czy wiesz", wersja 1.1},	
 author = {Marci{\'n}czuk, Micha{\l} and Piasecki, Dominik and Piasecki, Maciej and Radziszewski, Adam},	
 url = {http://hdl.handle.net/11321/39},	
 note = {{CLARIN}-{PL} digital repository},	
 year = {2013}	
}

"""


class PolishDyk(QuestionAnswering):

    """
    Class for German Quad Task
    """

    DATASET_ID = "allegro/klej-dyk"  
    TASK_NAME = "polish_dyk"
    HOMEPAGE_URL = "https://huggingface.co/datasets/allegro/klej-dyk"
    LANGUAGE = "pl"

