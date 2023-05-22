from typing import List, Union
from . import *

# mapping task to class objects
TASK_REGISTRY = {
    "germeval2018": germeval2018.GermEval2018,
    "germeval2017": germeval2017.GermEval2017,
    "gnad10": gnad10.Gnad10,
    "german_ner_legal": german_ner.GermanNerLegal,
    "german_europarl": german_europarl.GermanEuroParl,
    "german_quad": german_quad.GermanQuad,
    "spanish_quad": spanish_quad.SpanishQuad,
    "wiki_cat_es": wiki_cat_es.WikiCatEs,
    "spanish_conll": spanish_conll.SpanishConll,
    "flue": flue.Flue,
    "spanish_ehealth": spanish_ehealth.SpanishEhealth,
    "szeged_ner": szeged_ner.SzegedNer,
    "polish_dyk": polish_dyk.PolishDyk,
    "mapa": mapa.Mapa,
    "eur_lux": eur_lux.EurLux,
    "ehealth_kd": ehealth_kd.EhealthKd,
    "rucola": rucola.Rucola,
    "klej_dyk": klej_dyk.KlejDyk,
    "croatian_sentiment": croatian_sentiment.CroatianSentiment,
    "finish_sentiment": finish_sentiment.FinishSentiment,
    "swedish_ner": swedish_ner.SwedishNer,
    "greek_legal": greek_legal.GreekLegal,
    "bulgarian_sentiment":  bulgarian_sentiment.BulgarianSentiment,
    "czech_subjectivity": czech_subjectivity.CzechSubjectivity,
    "danish_misogyny": danish_misogyny.DanishMisogyny,
    "slovak_sentiment": slovak_sentiment.SlovakSentiment,
    "maltese_sentiment": maltese_sentiment.MalteseSentiment,
    "dutch_social": dutch_social.DutchSocial,
    "eur_lux_de": eur_lux.EurLuxDe,
    "eur_lux_en": eur_lux.EurLuxEn,
    "eur_lux_fr": eur_lux.EurLuxFr,
    

}

# mapping task to type
TASK_TYPE_REGISTRY = {
    "germeval2018": "classification",
    "germeval2017": "classification",
    "gnad10": "classification",
    "german_ner_legal": "ner",
    "german_europarl": "ner",
    "german_quad": "qa",
    "spanish_quad": "qa",
    "wiki_cat_es": "classification",
    "spanish_conll": "ner",
    "flue": "classification",
    "spanish_ehealth": "ner",
    "szeged_ner": "ner",
    "polish_dyk": "qa",
    "mapa": "ner",
    "eur_lux": "classification",
    "ehealth_kd": "ner",
    "rucola": "classification",
    "klej_dyk": "classification",
    "croatian_sentiment": "classification",
    "finish_sentiment": "classification",
    "swedish_ner": "ner",
    "greek_legal": "classification",
    "bulgarian_sentiment": "classification",
    "czech_subjectivity": "classification",
    "danish_misogyny": "classification",
    "slovak_sentiment": "classification",
    "maltese_sentiment": "classification",
    "dutch_social": "classification",
    "eur_lux_de": "classification",
    "eur_lux_en": "classification",
    "eur_lux_fr": "classification",
    
}

ALL_TASKS = sorted(list(TASK_REGISTRY))
ALL_TASK_TYPES = sorted(list(TASK_TYPE_REGISTRY))

'''
# returning task class
def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        print("Available tasks:")
        print(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}") from exc

'''

# return string names of all the tasks for reference
def get_all_tasks():
    all_task_str = []
    for key in TASK_REGISTRY:
        all_task_str.append(key)

    return all_task_str

def get_all_task_types():
    all_task_str = {}
    for key in TASK_TYPE_REGISTRY:
        all_task_str[key] = TASK_REGISTRY[key]
    
    return all_task_str

def get_dataset_information(dataset_name):
    #task_obj = TASK_REGISTRY[dataset_name]
    output_dict = []
    output_dict.append(TASK_REGISTRY[dataset_name]().get_label_name())
    output_dict.append(TASK_REGISTRY[dataset_name]().get_dataset_id())
    output_dict.append(TASK_REGISTRY[dataset_name]().get_url())

    return output_dict