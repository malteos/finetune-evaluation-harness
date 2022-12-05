import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import argparse
import json

from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.embeddings.base import TransformerEmbedding
#from flair.trainers import ModelTrainer
from trainer import ModelTrainer
package = "flair.datasets"

## Script to be used for fine-tuning of Classification Task

parser = argparse.ArgumentParser(
    description='Script to run evaluation on fine-tuned lm for text classification task.'
)
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--output_path', type = str, required= True, help='path for saving training logs and the final model')
parser.add_argument('--pooling_type', type = str, default = 'cls', help='pooling type for the created embedding')
parser.add_argument('--dataset_name', type = str, required=True, nargs = '*', help='please specify atleast one dataset')
parser.add_argument('--mini_batch_size', type = int, default=8)
parser.add_argument('--max_epochs', type = int, default = 60)
parser.add_argument('--learning_rate', default=3e-5)
parser.add_argument('--downsample', type=float, help='Do you want to downsample the corpus (specify a float value eg 0.1)')
parser.add_argument('--hidden_size', default = 64)
parser.add_argument(
    '--label_type', type = str,
    required = True,
    nargs='*',
    help='type of label eg class or sentiment. Specify one relevant label_type for each dataset in the same order in which dataset param is called.'
)
parser.add_argument('--do_full_train', type = bool, default = True,
    help='Allow for changes in transformer weights including classifcation layer. If set to False it would allow only classification layer to be adjusted'
    )
parser.add_argument('--save_results', type = bool, default = False, help = 'Do you want to save the results')

run_args = parser.parse_args()
print(run_args)

dataset_name = run_args.dataset_name
all_label_types = run_args.label_type           #list recording the label_types for each of the dataset

if(type(dataset_name) == list):
    for index, each_dataset in enumerate(dataset_name):
        ds_class = getattr(__import__(package, fromlist=[each_dataset]), each_dataset)
        if(run_args.downsample is not None):
            corpus = ds_class().downsample(run_args.downsample)
        else:
            corpus = ds_class()
        print(type(corpus))

        label_dict = corpus.make_label_dictionary(label_type= all_label_types[index])
        print(label_dict)

        document_embeddings = TransformerDocumentEmbeddings(
            model=run_args.model_name_or_path,
            cls_pooling = run_args.pooling_type,
            fine_tune=run_args.do_full_train,
            use_context=True,
        )

        document_embeddings.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(document_embeddings.tokenizer.pad_token)

        classifer = TextClassifier(
            document_embeddings,
            label_dictionary=label_dict,
            label_type = all_label_types[index]
        )

        trainer = ModelTrainer(classifer, corpus)

        fine_tune_res = trainer.fine_tune(
            run_args.output_path,
            learning_rate=run_args.learning_rate,
            mini_batch_size=run_args.mini_batch_size,
            max_epochs=run_args.max_epochs,
        )

        print(fine_tune_res)

        if(run_args.save_results is True):
            file_dest = str(run_args.output_path) + '/results.json'
            with open(file_dest, 'w', encoding='utf-8') as f:
                json.dump(fine_tune_res, f, ensure_ascii=False, indent=4)