"""Prediction module for real-world job posts"""

import pandas as pd
import joblib
from src.features.ruleset_features import extract_ruleset_features
from src.data.preprocessing import preprocess_text

JOB_TYPES = {
    0: "✅ Real Job",
    1: "⚠️ Identity Theft Scam",
    2: "⚠️ Fake Company / Impersonation",
    3: "⚠️ MLM / Pyramid Scheme"
}

JOB_REASONS = {
    0: "Looks professional with normal requirements.",
    1: "Asks for personal information or promises easy money.",
    2: "Pretends to represent a well-known company.",
    3: "Focuses on recruiting others and passive income."
}

def predict_job(job, model_path=None, model=None, vectorizer=None):
    """
    Predict the type of job posting

    Args:
        job (dict): Job posting with keys: title, company_profile, description,
                   requirements, benefits
        model_path (str): Path to saved model file
        model: Pre-loaded model (optional)
        vectorizer: Pre-loaded vectorizer (optional)

    Returns:
        tuple: (prediction, confidence)
    """
    # Combine text fields
    text = f"{job['company_profile']} {job['description']} {job['requirements']} {job['benefits']}"

    # Create DataFrame
    df = pd.DataFrame([{
        'title': job['title'],
        'company_profile': job['company_profile'],
        'description': job['description'],
        'requirements': job['requirements'],
        'benefits': job['benefits'],
        'combined_text': text
    }])

    # Apply preprocessing
    df['processed_text'] = df['combined_text'].apply(preprocess_text)

    # Extract ruleset features
    ruleset_features_df = extract_ruleset_features(df)
    if 'type' in ruleset_features_df.columns:
        ruleset_features_df = ruleset_features_df.drop('type', axis=1)

    # Load model if path provided
    if model_path and not model:
        model = joblib.load(model_path)

    if not model:
        raise ValueError("Either model_path or model must be provided")

    # Make prediction
    features_combined = ruleset_features_df.values
    prediction = model.predict(features_combined)[0]

    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(features_combined)[0])
    else:
        confidence = 1.0

    return prediction, confidence

def get_job_info(prediction):
    """Get human-readable information about prediction"""
    return {
        'type': JOB_TYPES.get(prediction, "Unknown"),
        'reason': JOB_REASONS.get(prediction, "Unable to classify"),
        'recommendation': "Appears safe, still verify employer." if prediction == 0 else "Do NOT apply without verification."
    }