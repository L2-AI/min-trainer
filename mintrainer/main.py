from functools import cache
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from _utils import (
    get_id2label_and_label2id,
    dataframe_to_tokenized_dataset,
    compute_metrics,
)

DEFAULT_TRAINING_ARGS = TrainingArguments(
    output_dir=None,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=5,
    bf16=False,
    optim="adamw_torch",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    report_to=[],
)

def MinTrainer(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categories: list[str],
        model_name:str ='answerdotai/ModernBERT-base',
        cache_dir:str = None,
        finetuned_model_output_dir:str = None,
        training_args:TrainingArguments = DEFAULT_TRAINING_ARGS
    ) -> pd.DataFrame:
    
    if finetuned_model_output_dir:
        training_args.output_dir = finetuned_model_output_dir

    id2label, label2id = get_id2label_and_label2id(categories)

    model = ModernBertForSequenceClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        cache_dir=cache_dir
    )
    if torch.cuda.is_available():
        model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    tokenized_train_dataset = dataframe_to_tokenized_dataset(train_df, tokenizer)
    tokenized_test_dataset = dataframe_to_tokenized_dataset(test_df, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    return