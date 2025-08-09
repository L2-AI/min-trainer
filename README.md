# MinTrainer

MinTrainer is a light-weight pipeline for fine-tuning [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) models for classification tasks. This project was designed for settings where one has multiple datasets containing observations with semantic meaning (e.g. text snippets, tokens) + associated categorical labels, and it is performant to train a unique model for each dataset (as is often the case in data mapping).

This code was written to be compatible with [Google Colab](https://colab.research.google.com/), where free T4 GPUs are available. Using MinTrainer with a T4 GPU reduces training time to minutes for datasets with 20,000 input samples of 100 characters (5 epochs, training loss reduced to approx. 0).

This open-source project was created by and is property of [L2 Labs](https://l2labs.ai/) but is free to use without restrictions.

## Installation

Clone the repostiory:

```
git clone https://github.com/L2-AI/min-trainer.git
```

From the root directory `min-trainer/` run:

```
pip install e .
```

## Usage

To train a ModernBERT model on four categories from the `sklearn` `20newsgroups` dataset, run the following:

```Python
from mintrainer._sample_dataset_utils import get_newsgroup_df_and_targets
from mintrainer.main import MinTrainer


train_df, categories = get_newsgroup_df_and_targets(subset='train')
test_df, _ = get_newsgroup_df_and_targets(subset='test')

MinTrainer(
    train_df = train_df, # pd.DataFrame with 2 columns: `text` (as str) and `labels` (as int)
    test_df = test_df, # same as above
    categories=categories, # ordered list of label text ( `categories[0]` should map to label `0`)
    #cache_dir= # dir where HuggingFace transformers saves and reads models
    #finetuned_model_output_dir= # dir where finetuned model checkpoints are saved
)
```

Or from the directory `min-trainer/examples/` run the following from the CLI:

```bash
python newsgroup_classification.py
```
