import torch
from transformers import AutoTokenizer
import pandas as pd


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

df = pd.read_csv('data.csv')
df['text'] = df['Messages'].fillna('') + " " + df['Remain Messages'].fillna('')

prepared_data = {"text": df['text'].tolist()}

tokenized_data = tokenize_function(prepared_data)