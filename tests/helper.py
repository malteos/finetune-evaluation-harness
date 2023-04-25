import os
from transformers import HfArgumentParser, TrainingArguments, BertConfig, BertModel


custom_config = BertConfig(n_layers=2, n_heads=2, n_positions=2, n_emd = 2, hidden_size=12, max_position_embeddings=4, vocab_size=16)
print(custom_config)
custom_model = BertModel(custom_config)
custom_model.save_pretrained("custom_model")

