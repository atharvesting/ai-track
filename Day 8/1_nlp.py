'''
Based on Getting started with NLP for absolute beginners (kaggle)
https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners

The objective is to phrase match, i.e. take two phrases and determine how similar they are in meaning.

The dataset provides the data necessary to evaluate this -
- Anchor: The base phrase (one of the two phrases)
- Target: The phrase to be compared with the Anchor
- Context: Category/Topic the phrases belong to

'''
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer

df = pd.read_csv('data/Phrase Matching/train.csv')

# print(df.describe(include='object'))  # In order to include text-based columns as well

'''
In order to perform this task, we need to formulate the information in a form that the
NLP model can understand. Since they are usually designed to process a single string of 
text as input, we can combine the anchor, target and context into one string and store this
input string in the DataFrame itself.

TEXT1, TEXT2 and ANC1 are special markers that are used as separators and labels so that 
the NLP model can effectively interpret and distinguish between the information, leading to 
better performance.

'''
df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor

ds = Dataset.from_pandas(df)  # NLP transformers use Dataset objects

'''
Since a deep learning model expects numbers as input, we must engage in two procedures to prepare
the data for processing -

- Tokenization: Split a string into words/tokens
- Numericalization: Convert each token into a number

'''
model_nm = 'distilbert-base-uncased'
tokz = AutoTokenizer.from_pretrained(model_nm)

# print(tokz.tokenize("Hello my name is X and I am learning Natural Language Processing"))

def tok_func(x): return tokz(x['input'])
tok_ds = ds.map(tok_func, batched=True)

# row = tok_ds[0]
# print(row['input'], row['input_ids'])

tok_ds = tok_ds.rename_columns({'score':'labels'})

dds = tok_ds.train_test_split(0.25, seed=42)  # 25% val, 75% train

eval_df = pd.read_csv('data/Phrase Matching/test.csv')
eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

bs = 128 # Batch size
epochs = 4

args = TrainingArguments('outputs', learning_rate=2e-5, warmup_ratio=0.1, lr_scheduler_type='cosine',
                         per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
                         num_train_epochs=epochs, weight_decay=0.01,
                         report_to='none')

trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'])

trainer.train()

