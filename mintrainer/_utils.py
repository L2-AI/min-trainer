from typing import Tuple
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import TrainingArguments

def get_id2label_and_label2id(ordered_label_names: list) -> Tuple[dict, dict]:
  id2label = {k:v for (k,v) in zip(range(0,len(ordered_label_names)), ordered_label_names)}
  label2id = {k:v for (k,v) in zip(id2label.values(), id2label.keys())}

  return id2label, label2id


def dataframe_to_tokenized_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    expected_columns = {'text', 'labels'}
    if set(df.columns) != expected_columns:
        raise ValueError(f"DataFrame must have exactly two columns: {expected_columns}. Found: {set(df.columns)}")

    dataset = Dataset.from_pandas(df)
    
    tokenized_dataset = dataset.map(
        lambda batch: tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt"),
        batched=True,
        batch_size=(dataset.shape[0]+1),
        remove_columns=["text"]
    )
    
    return tokenized_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    return {
        'f1': f1_score(labels, predictions, average='weighted'),
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }