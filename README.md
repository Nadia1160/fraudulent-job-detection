# Fraudulent Job Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://kaggle.com)

A machine learning system for detecting fraudulent job postings using NLP and ensemble methods.

## 📊 Dataset
- **Source**: EMSCAD (Employment Scam Dataset)
- **Samples**: ~17,880 job postings
- **Classes**: Real Job, Identity Theft, Corporate Identity Theft, MLM

## 🚀 Quick Start
```python
from src.prediction import FraudJobDetector

detector = FraudJobDetector('models/best_model.pkl')
result = detector.predict(job_posting)
print(f"Prediction: {result['type']}") 