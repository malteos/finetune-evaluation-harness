import os
import sys


from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


@dataclass
class RunArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        metadata={"help": "The path of the dataset to use (via the datasets library)."}
    )
    model_name_or_path: str = field()

    output_path: str = field()
    


def main():
    parser = HfArgumentParser((RunArguments, ))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        run_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        run_args = parser.parse_args_into_dataclasses()

    run_args = run_args[0]

    # 1. get the corpus
    corpus = CONLL_03()
    print(corpus)

    # 2. what label do we want to predict?
    label_type = 'ner'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
    print(label_dict)

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(model='xlm-roberta-large',
                                        layers="-1",
                                        subtoken_pooling="first",
                                        fine_tune=True,
                                        use_context=True,
                                        )

    # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type='ner',
                            use_crf=False,
                            use_rnn=False,
                            reproject_embeddings=False,
                            )

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. run fine-tuning
    trainer.fine_tune('resources/taggers/sota-ner-flert',
                    learning_rate=5.0e-6,
                    mini_batch_size=4,
                    mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
                    )