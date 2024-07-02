import torch
from transformers import DistilBertTokenizer, DistilBertModel

def load_distilbert_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model
