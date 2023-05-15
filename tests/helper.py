import os
from transformers import HfArgumentParser, TrainingArguments, BertConfig, BertModel


custom_config = BertConfig(n_layers=2, n_heads=2, num_hidden_layers=2, hidden_size=12)
print(custom_config)
custom_model = BertModel(custom_config)
custom_model.save_pretrained("custom_model")

