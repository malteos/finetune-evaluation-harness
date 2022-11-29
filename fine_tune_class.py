import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import argparse

from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.embeddings.base import TransformerEmbedding
from flair.trainers import ModelTrainer
package = "flair.datasets"


## Script to be used for fine-tuning of Classification Task

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--output_path', type = str, required= True)
parser.add_argument('--pooling_type', type = str, default = 'first_last')
parser.add_argument('--dataset_name', type = str, default = 'GERMEVAL_2018_OFFENSIVE_LANGUAGE')
parser.add_argument('--mini_batch_size', type = int, default=32)
parser.add_argument('--max_epochs', type = int, default = 15)
parser.add_argument('--learning_rate', default=3e-5)
parser.add_argument('--hidden_size', default = 64)
parser.add_argument('--label_type', type = str, default='class')

run_args = parser.parse_args()

name = run_args.dataset_name
ds_class = getattr(__import__(package, fromlist=[name]), name)
corpus = ds_class()
print(corpus)
print(name)
print(run_args.model_name_or_path)
print(run_args.output_path)
print(run_args.pooling_type)

label_dict = corpus.make_label_dictionary(label_type= run_args.label_type)
print(label_dict)

document_embeddings = TransformerDocumentEmbeddings(
    model=run_args.model_name_or_path,
    cls_pooling = 'cls',
    fine_tune=True,
    use_context=True,
)


document_embeddings.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print(document_embeddings.tokenizer.pad_token)

classifer = TextClassifier(
    document_embeddings,
    label_dictionary=label_dict,
    label_type = run_args.label_type
)

trainer = ModelTrainer(classifer, corpus)

fine_tune_res = trainer.fine_tune(
    run_args.output_path,
    learning_rate=run_args.learning_rate,
    mini_batch_size=run_args.mini_batch_size,
    max_epochs=run_args.max_epochs
)

print(fine_tune_res)