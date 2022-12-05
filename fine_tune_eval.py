import os
import sys, flair
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from flair.data import MultiCorpus
import argparse
import json
from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings, SentenceTransformerDocumentEmbeddings
#from flair.models import SequenceTagger, TextClassifier
from sequence_tagger import SequenceTagger

#from flair.trainers import ModelTrainer
from trainer import ModelTrainer
package = "flair.datasets"

## Script to be used for fine-tuning of NER Task

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--output_path', type = str, required= True, help='path for saving training logs and the final model')
parser.add_argument('--pooling_type', type = str, default = 'mean', help='pooling type for the created embedding')
parser.add_argument('--dataset_name', type = str, required=True, nargs = '*', help='please specify atleast one dataset')
parser.add_argument('--mini_batch_size', type = int, default=8)
parser.add_argument('--max_epochs', type = int, default = 15)
parser.add_argument('--learning_rate', default=3e-5)
parser.add_argument('--hidden_size', default = 64)
parser.add_argument('--label_type', type = str, default='ner', help='type of label eg ner (default)')
parser.add_argument('--downsample', type=float, help='Do you want to downsample the overall corpus (specify a float value eg 0.1)')
parser.add_argument('--do_full_train', type = bool, default = True,
    help='Allow for changes in transformer weights including classifcation layer. If set to False it would allow only classification layer to be adjusted'
    )
parser.add_argument("-c", "--cuda", type=int, default=0, help="CUDA device")
parser.add_argument('--save_results', type = bool, default = False, help = 'Do you want to save the results')


run_args = parser.parse_args()
print(run_args)
flair.device = f'cuda:{str(run_args.cuda)}'


dataset_name = run_args.dataset_name
ds_class_list = []
for each_dataset in dataset_name:
    ds_class = getattr(__import__(package, fromlist=[each_dataset]), each_dataset)
    if run_args.downsample is not None:
        ds_class_list.append(ds_class().downsample(run_args.downsample))
    else:
        ds_class_list.append(ds_class())

corpus = MultiCorpus(ds_class_list)


label_type = run_args.label_type
label_dict = corpus.make_label_dictionary(label_type=label_type)

print(label_dict)

embeddings = TransformerWordEmbeddings(
    model=run_args.model_name_or_path,
    layers="-1",
    subtoken_pooling=run_args.pooling_type,
    fine_tune=run_args.do_full_train,
    use_context=True,
    allow_long_sentences = True,
    embeddings_storage_mode = 'gpu',
    respect_document_boundaries=True,
)

print(type(embeddings))
embeddings.tokenizer.pad_token = embeddings.tokenizer.eos_token
embeddings.tokenizer.bos_token = embeddings.tokenizer.eos_token
embeddings.tokenizer.unk_token = embeddings.tokenizer.eos_token
embeddings.force_max_length = True
embeddings.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print(embeddings.tokenizer.pad_token)
print(embeddings.embedding_length)
print(embeddings.context_length)


tagger = SequenceTagger(
    hidden_size=run_args.hidden_size,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type= run_args.label_type,
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False
)


trainer = ModelTrainer(tagger, corpus)

fine_tune_res = trainer.fine_tune(
    run_args.output_path,
    learning_rate=run_args.learning_rate,
    mini_batch_size=run_args.mini_batch_size,
    mini_batch_chunk_size = 32,
    max_epochs=run_args.max_epochs,
)

print(fine_tune_res)

if(run_args.save_results is True):
    file_dest = str(run_args.output_path) + '/results.json'
    with open(file_dest, 'w', encoding='utf-8') as f:
        json.dump(fine_tune_res, f, ensure_ascii=False, indent=4)