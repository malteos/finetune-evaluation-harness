"""

The Czy wiesz? (eng. Did you know?) the dataset consists of almost 5k question-answer pairs obtained from Czy wiesz... 
section of Polish Wikipedia. Each question is written by a Wikipedia collaborator and is answered with a link to a relevant Wikipedia article. 
In huggingface version of this dataset, they chose the negatives which have the largest token overlap with a question.


Tasks (input, output, and metrics)
The task is to predict if the answer to the given question is correct or not.

Input ('question sentence', 'answer' columns): question and answer sentences

Output ('target' column): 1 if the answer is correct, 0 otherwise.

Domain: Wikipedia

Measurements: F1-Score

Example:

Input: Czym zajmowali się świątnicy? ; Świątnik – osoba, która dawniej zajmowała się obsługą kościoła (świątyni).

Input (translated by DeepL): What did the sacristans do? ; A sacristan - a person who used to be in charge of the handling the church (temple).

Output: 1 (the answer is correct)


"""


from .classification import Classification


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



class KlejDyk(Classification):

    """
    Class for Klej Dyk Classification Task
    """


    DATASET_ID = "allegro/klej-dyk"  
    TASK_NAME = "klej_dyk"
    LABEL_NAME = "target"  
    HOMEPAGE_URL = "https://huggingface.co/datasets/allegro/klej-dyk"
    LANGUAGE = "pl"
    
