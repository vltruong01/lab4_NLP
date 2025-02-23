# lab4_NLP
![alt text](image-2.png)

Task1:
Dataset Description
The dataset used for training BERT in this notebook is a subset of the Wikipedia dataset. 
The Wikipedia dataset is a large-scale dataset containing text from Wikipedia articles. 
It is commonly used for natural language processing (NLP) tasks, 
including language modeling and pre-training of transformer models like BERT.

Dataset Source
The Wikipedia dataset is sourced from Hugging Face's datasets library. 
Hugging Face provides a wide range of datasets for various NLP tasks, 
and the Wikipedia dataset is one of the most popular datasets for pre-training 
language models.

https://huggingface.co/datasets/legacy-datasets/wikipedia

Task3: 
![alt text](image.png)
![alt text](image-1.png)

Challenges:
Small Batch Size: Leads to noisy gradient updates and slower convergence.
Limited Training Epochs: One epoch restricts generalization.
Dataset Imbalance: Unequal class distribution may impact learning.

Improvements:
Hyperparameter Tuning: Adjust learning rates, batch sizes, and optimizers.
Larger Model: Using BERT-base-uncased for better representations.
More Training Epochs: Fine-tuning longer with learning rate scheduling.
Addressing Imbalance: Oversampling or data augmentation.
Enhanced Representations: Using CLS token or self-attention instead of mean pooling.
