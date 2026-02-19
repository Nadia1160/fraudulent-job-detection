# Fraudulent Job Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://kaggle.com)

A machine learning system for detecting fraudulent job postings using NLP techniques. This project implements the methodology described in the paper "A Machine Learning Approach to Detecting Fraudulent Job Types" (AI & SOCIETY, 2023).

## 📊 Dataset
- **Source**: EMSCAD (Employment Scam Dataset)
- **Samples**: ~17,880 job postings
- **Classes**: Real Job, Identity Theft, Corporate Identity Theft, MLM

## 📋 Overview

This project classifies job postings into four categories:
- **Type 0**: Real job (non-fraudulent)
- **Type 1**: Identity Theft Scam
- **Type 2**: Fake Company / Impersonation
- **Type 3**: MLM / Pyramid Scheme

## 🚀 Features

- **Multi-class Classification**: Identifies 4 types of job postings
- **Multiple Feature Sets**:
  - Bag of Words & TF-IDF (6356 features)
  - Empirical Ruleset Features (based on linguistic analysis)
  - Word2Vec Embeddings (100 dimensions)
  - Transformer Models (BERT/RoBERTa)
  - Combined feature sets
- **8 Classifiers**: LR, SGD, KNN, CART, SVM, RF, AB, GB
- **Transformer Support**: BERT, RoBERTa, DistilBERT

## 📊 Results

Best performing model: **Ruleset+POS+BoW with Logistic Regression**
- Weighted Avg. F1-Score: 0.9157
- Accuracy: 0.9167
- MCC: 0.8836

## 🚀 Quick Start
```python
from src.prediction import FraudJobDetector

detector = FraudJobDetector('models/best_model.pkl')
result = detector.predict(job_posting)
print(f"Prediction: {result['type']}") 