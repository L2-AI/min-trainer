# MinTrainer

MinTrainer is a light-weight pipeline for fine-tuning [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) models for classification tasks. This project is most useful in settings where there are many datasets containing observations with semantic meaning (e.g. text snippets, tokens) + associated categorical labels, and it is performant to train a unique model for each dataset (as is often the case in data mapping).

This code was written to be compatible with [Google Colab](https://colab.research.google.com/), where free T4 GPUs are available. Using MinTrainer with a T4 GPU reduces training time to minutes for datasets with 20,000 input samples of 100 characters (5 epochs, training loss reduced to approx. 0).

This open-source project was created by and is property of [L2 Labs](https://l2labs.ai/) but is free to use without restrictions.
