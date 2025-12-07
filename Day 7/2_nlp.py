import pandas as pd
from datasets import Dataset, DatasetDict

df = pd.read_csv('data/Phrase Matching/train.csv')



df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor

ds = Dataset.from_pandas(df)

model_nm = 'microsoft/deberta-v3-small'

from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)

print(tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!"))