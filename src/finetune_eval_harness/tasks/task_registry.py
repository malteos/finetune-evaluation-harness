from typing import List, Union

from . import (
    germeval2018,
    germeval2017,
    gnad10,
    german_ner,
    german_europarl,
    german_quad,
    multi_eurlex,
    spanish_quad,
    wiki_cat_es,
    spanish_conll,
    flue,
    spanish_ehealth,
    szeged_ner,
    polish_dyk,
    mapa,
    ehealth_kd,
    rucola,
    klej_dyk,
    croatian_sentiment,
    finish_sentiment,
    swedish_ner,
    greek_legal,
    bulgarian_sentiment,
    czech_subjectivity,
    danish_misogyny,
    slovak_sentiment,
    maltese_sentiment,
    dutch_social,
    piaf,
    xquad,
    pawsx,
    xnli,
    xstance,
    conll2003,
    xtreme_panx_ner,
)

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
    "ehealth_kd": ehealth_kd.EhealthKd,
    "rucola": rucola.Rucola,
    "klej_dyk": klej_dyk.KlejDyk,
    "croatian_sentiment": croatian_sentiment.CroatianSentiment,
    "finish_sentiment": finish_sentiment.FinishSentiment,
    "swedish_ner": swedish_ner.SwedishNer,
    "greek_legal": greek_legal.GreekLegal,
    "bulgarian_sentiment": bulgarian_sentiment.BulgarianSentiment,
    "czech_subjectivity": czech_subjectivity.CzechSubjectivity,
    "danish_misogyny": danish_misogyny.DanishMisogyny,
    "slovak_sentiment": slovak_sentiment.SlovakSentiment,
    "maltese_sentiment": maltese_sentiment.MalteseSentiment,
    "dutch_social": dutch_social.DutchSocial,
    "multi_eurlex_en": multi_eurlex.MultiEURLexEN,
    "multi_eurlex_de": multi_eurlex.MultiEURLexDE,
    "multi_eurlex_fr": multi_eurlex.MultiEURLexFR,
    "multi_eurlex_es": multi_eurlex.MultiEURLexES,
    "multi_eurlex_it": multi_eurlex.MultiEURLexIT,
    "piaf": piaf.Piaf,
    # "mapa": mapa.Mapa,
    "mapa_de": mapa.MapaDe,
    "mapa_en": mapa.MapaEn,
    "mapa_fr": mapa.MapaFr,
    # "xquad": xquad.XQuad,
    # "xquad_de": xquad.XQuadDe,
    # "xquad_en": xquad.XQuadEn,
    # "xquad_es": xquad.XQuadEs,
    "pawsx_de": pawsx.PawsXDe,
    "pawsx_en": pawsx.PawsXEn,
    "paws_es": pawsx.PawsXEs,
    "xnli_de": xnli.XnliDe,
    "xnli_es": xnli.XnliEs,
    "xnli_en": xnli.XnliEn,
    "xstance_fr": xstance.XStanceFR,
    "xstance_de": xstance.XStanceDE,
    # "xstance_it": xstance.XStanceIT,
    "conll2003": conll2003.Conll2003,
    "xtreme_panx_ner_en": xtreme_panx_ner.XtremePanxEN,
    "xtreme_panx_ner_de": xtreme_panx_ner.XtremePanxDE,
    "xtreme_panx_ner_fr": xtreme_panx_ner.XtremePanxFR,
    "xtreme_panx_ner_es": xtreme_panx_ner.XtremePanxES,
    "xtreme_panx_ner_it": xtreme_panx_ner.XtremePanxIT,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))
# ALL_TASK_TYPES = sorted(list(TASK_TYPE_REGISTRY))


# return string names of all the tasks for reference
def get_all_tasks():
    all_task_str = []
    for key in TASK_REGISTRY:
        all_task_str.append(key)

    return all_task_str


# def get_all_task_types():
#     all_task_str = {}
#     for key in TASK_TYPE_REGISTRY:
#         all_task_str[key] = TASK_REGISTRY[key]

#     return all_task_str


def get_dataset_information(dataset_name):
    # task_obj = TASK_REGISTRY[dataset_name]
    output_dict = []
    output_dict.append(TASK_REGISTRY[dataset_name]().get_label_name())
    output_dict.append(TASK_REGISTRY[dataset_name]().get_dataset_id())
    output_dict.append(TASK_REGISTRY[dataset_name]().get_url())

    return output_dict
