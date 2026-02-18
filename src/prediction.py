\"\"\"Fraudulent job prediction module.\"\"\"

import pandas as pd
import joblib
import re
from typing import Dict, Any, Tuple

class FraudJobDetector:
    \"\"\"Main class for detecting fraudulent job postings.\"\"\"
    
    def __init__(self, model_path: str):
        \"\"\"Initialize detector with trained model.\"\"\"
        self.model = joblib.load(model_path)
        self.bow_vectorizer = joblib.load('models/bow_vectorizer.pkl')
        
    def predict(self, job: Dict[str, str]) -> Dict[str, Any]:
        \"\"\"Predict if a job posting is fraudulent.\"\"\"
        # Preprocess text
        text = self._preprocess_text(job)
        
        # Extract features
        features = self._extract_features(job, text)
        
        # Predict
        prediction = self.model.predict(features)[0]
        confidence = max(self.model.predict_proba(features)[0])
        
        return {
            'type': int(prediction),
            'confidence': float(confidence),
            'label': self._get_label(prediction)
        }
    
    def _preprocess_text(self, job: Dict[str, str]) -> str:
        \"\"\"Preprocess job text.\"\"\"
        text = f"{job.get('company_profile', '')} {job.get('description', '')}"
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_features(self, job: Dict[str, str], text: str) -> pd.DataFrame:
        \"\"\"Extract features for prediction.\"\"\"
        # Simplified feature extraction
        features = pd.DataFrame([{
            'title_length': len(job.get('title', '')),
            'has_company': 1 if job.get('company_profile') else 0,
            'text_length': len(text)
        }])
        return features
    
    def _get_label(self, pred: int) -> str:
        \"\"\"Convert prediction to label.\"\"\"
        labels = {
            0: "Real Job",
            1: "Identity Theft Scam",
            2: "Fake Company",
            3: "MLM/Pyramid Scheme"
        }
        return labels.get(pred, "Unknown")
