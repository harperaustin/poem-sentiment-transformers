# Poetry Sentiment Classification with Transformers and Interpretability

This project uses a fine-tuned DistilBERT model to classify the sentiment of poetry excerpts into four categories: **Negative**, **No Impact**, **Positive**, and **Mixed**. It leverages the Hugging Face Transformers library for training and LIME for interpretability.

> **Goal:** Build a working NLP classifier using a transformer architecture and gain deeper insight into how transformer models make predictions through interpretability tools.

---

## Project Highlights

-  Fine-tuned a pre-trained `distilbert-base-uncased` transformer on the [`poem_sentiment`](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) dataset
-  Achieved **86% validation accuracy**
-  Used **LIME** to visualize how individual words influence the model's predictions
-  Interpreted model weaknesses (e.g. failure to detect negation in certain phrases like _“I am not sad”_)
-  Developed an understanding of how transformer models are adapted for classification (vs. next-token prediction)

---

## What I Learned

- How to fine-tune a transformer model (DistilBERT) for a custom text classification task
- How transformer-based models like BERT process sentences and output classification logits
- How to apply interpretability techniques (LIME) to analyze what words influenced a prediction
- How to evaluate model performance and explore common NLP failure cases

---

## Dataset

- Source: [`google-research-datasets/poem_sentiment`](https://huggingface.co/datasets/google-research-datasets/poem_sentiment)
- Dataset contains short poetry snippets labeled as:
  - **Negative**
  - **No Impact**
  - **Positive**
  - **Mixed**
- Only ~900 labeled examples in the training set, which makes this a challenging low-data fine-tuning task

---

## Tech Stack

- Python
- Hugging Face Transformers (`AutoModelForSequenceClassification`, `Trainer`)
- PyTorch
- LIME (Local Interpretable Model-Agnostic Explanations)
- Jupyter Notebook

---

## Model Limitations

- Struggles with negations/nuanced context around words.
- Trained on a relatively small dataset.

---

## Future Work

- Data augmentation to address small dataset
- Improve handling of negations/complex sentence structure
- Using other transformer architectures



