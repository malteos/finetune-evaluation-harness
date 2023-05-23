"""
The dataset contains 10 files with around 271,342 tweets. 
The tweets are filtered via the official Twitter API to contain tweets 
in Dutch language or by users who have specified their location information within Netherlands geographical boundaries. 
Using natural language processing we have classified the tweets 
for their HISCO codes. If the user has provided their location within Dutch boundaries, 
we have also classified them to their respective provinces The objective of 
this dataset is to make research data available publicly 
in a FAIR (Findable, Accessible, Interoperable, Reusable) way. Twitter's 
Terms of Service Licensed under Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (2020-10-27)


"""

from .classification import Classification


_DESCRIPTION = """
Classification of social media data in dutch language (See more for data collection from the web page url) 
"""


_CITATION = """
@data{FK2/MTPTL7_2020, author = {Gupta, Aakash}, publisher = {COVID-19 Data Hub}, title = {{Dutch social media collection}}, year = {2020}, version = {DRAFT VERSION}, doi = {10.5072/FK2/MTPTL7}, url = {https://doi.org/10.5072/FK2/MTPTL7} }
"""


class DutchSocial(Classification):


    DATASET_ID = "dutch_social"  # HF datasets ID
    TASK_NAME = "dutch_social"
    LABEL_NAME = "label"  # column name from HF dataset
    HOMEPAGE_URL = "https://huggingface.co/datasets/dutch_social"
    LANGUAGE = "nl"