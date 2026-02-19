"""Model training module"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, matthews_corrcoef, classification_report)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from tqdm import tqdm
import joblib
from typing import Optional, Any, Dict, List, Tuple

def get_classifiers():
    """Return list of classifiers as per paper"""
    return [
        ('LR', LogisticRegression(max_iter=1000, random_state=42)),
        ('SGD', SGDClassifier(random_state=42)),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier(random_state=42)),
        ('SVM', SVC(random_state=42, probability=True)),
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('AB', AdaBoostClassifier(n_estimators=100, random_state=42)),
        ('GB', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]

def evaluate_feature_set(X, y, feature_name, classifiers_list):
    """
    Evaluate a feature set with all classifiers

    Args:
        X: Feature matrix
        y: Target labels
        feature_name (str): Name of feature set
        classifiers_list (list): List of (name, classifier) tuples

    Returns:
        pd.DataFrame: Results dataframe
    """
    print(f"\n📊 Evaluating: {feature_name}")
    print(f"   Samples: {len(X)}, Features: {X.shape[1]}")

    # Split data (80/20 stratified as per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
    )

    results = []

    for clf_name, clf in tqdm(classifiers_list, desc="Training"):
        try:
            model = clone(clf)

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Get per-class metrics
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            results.append({
                'Feature Set': feature_name,
                'Classifier': clf_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'MCC': mcc,
                'Model': model,
                'y_test': y_test,
                'y_pred': y_pred,
                'Class_Report': class_report
            })

        except Exception as e:
            print(f"   Error with {clf_name}: {str(e)[:50]}")
            continue

    return pd.DataFrame(results)

class FraudJobClassifier:
    """Fraudulent Job Classifier"""

    def __init__(self, model_type='lr', feature_set='combined_bow', random_state=42):
        """
        Initialize classifier

        Args:
            model_type (str): Type of classifier ('lr', 'rf', 'svm', etc.)
            feature_set (str): Feature set name
            random_state (int): Random seed
        """
        self.model_type = model_type
        self.feature_set = feature_set
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.classifiers = dict(get_classifiers())

        if model_type in self.classifiers:
            self.model = self.classifiers[model_type]
        else:
            raise ValueError(f"Model type {model_type} not recognized")

    def fit(self, X, y):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not initialized or trained")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not initialized or trained")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Model does not support predict_proba")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not initialized or trained")
            
        y_pred = self.predict(X_test)

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }

        return results

    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Load model from file"""
        self.model = joblib.load(filepath)