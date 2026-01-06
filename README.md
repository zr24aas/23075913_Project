# SMS Spam / Non-Spam Detection using Machine Learning and NLP

# Project Overview
This project investigates and compares the effectiveness of two different approaches for **SMS spam detection**:

1. **Multinomial Naive Bayes** – a traditional machine learning model developed from scratch.
2. **BERT (Bidirectional Encoder Representations from Transformers)** – a pre-trained transformer-based deep learning model fine-tuned for text classification.

The objective of this project is to evaluate how models developed from scratch compare with pre-trained models in terms of classification performance, computational cost, and practical applicability for SMS spam and non-spam detection.

---

# Research Question
How effectively do models developed from scratch and pre-trained models classify SMS messages as spam or non-spam?

---

# Dataset
- **Name:** SMS Phishing Dataset for Machine Learning and Pattern Recognition  
- **Source:** Mendeley Data  
- **Authors:** Mishra, S. and Soni, D. (2022)  
- **Link:** https://data.mendeley.com/datasets/f45bkkt8pr/1  

The dataset contains **5,971 anonymised SMS messages** labelled as *ham*, *spam*, or *smishing*.  
For this project, spam and smishing messages were merged into a single **spam** class to formulate a binary classification task (spam vs non-spam).

The dataset is publicly available and suitable for academic research.

---

# Data Pre-processing
- Converted all text and labels to lowercase
- Removed non-text columns (URL, EMAIL, PHONE)
- Cleaned and normalised text content
- Applied stemming for the Naive Bayes model
- Used **TF-IDF vectorisation** for Naive Bayes
- Applied **tokenisation** for BERT (no manual feature engineering)

---

# Models Implemented

# 1. Multinomial Naive Bayes
- Feature extraction using **TF-IDF**
- Hyperparameter tuning using **GridSearchCV**
- Evaluated using accuracy, precision, recall, F1-score, and confusion matrix

# 2. BERT (Transformer-based Model)
- Fine-tuned a pre-trained BERT model
- Tokenisation with attention masks
- Optimised using AdamW optimiser and learning rate scheduling
- Evaluated using the same metrics as Naive Bayes

---

# Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

# Results Summary
- **Naive Bayes** achieved strong baseline performance with high overall accuracy but lower recall for spam messages due to its feature independence assumption.
- **BERT** achieved higher accuracy and F1-score, demonstrating improved detection of contextual and semantic spam patterns.
- BERT showed a better balance between false positives and false negatives, making it more suitable for real-world SMS spam detection scenarios.

---

# Technologies Used
- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- PyTorch  
- Transformers  

---

