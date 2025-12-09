import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer

model_nm = 'distilbert-base-uncased'  # Defining a pre-trained model
'''
Alternative:
- 'microsoft/deberta-v3-small' is a powerful, efficient model created by Microsoft.
    - Needs sentencepiece library to work.
'''

'''
tokz stores the vocabulary and rules necessary for doing tokenization.
- Vocabulary: Massive lookup table that maps text strings to unique integer IDs.
- Rules: Normalisation and Subword Splitting.

tokz.vocab_size = To check size of vocabulary
tokz.convert_tokens_to_ids(token) = To check the integer ID of a token
'''
tokz = AutoTokenizer.from_pretrained(model_nm, use_fast=False)
print(tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!"))
# OUTPUT: ['g', "'", 'day', 'folks', ',', 'i', "'", 'm', 'jeremy', 'from', 'fast', '.', 'ai', '!']

print(f"Vocabulary size == {tokz.vocab_size}, ID of 'hello' = {tokz.convert_tokens_to_ids('hello')}")