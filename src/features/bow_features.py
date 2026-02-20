"""Bag of Words and TF-IDF feature extraction"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import spmatrix
import numpy as np

def extract_bow_features(df, text_column='processed_text', max_features=6356):
    """
    Extract Bag of Words features

    Args:
        df (pd.DataFrame): DataFrame with text column
        text_column (str): Name of text column
        max_features (int): Maximum vocabulary size

    Returns:
        tuple: (df_bow, vectorizer)
    """
    bow_vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )

    bow_features = bow_vectorizer.fit_transform(df[text_column].astype(str).fillna(''))

    # Safe conversion of sparse matrix to dense
    if bow_features.shape[1] < 10000:
        # Check if it's a sparse matrix and convert safely
        if hasattr(bow_features, 'toarray'):
            dense_array = bow_features.toarray()  # type: ignore
        else:
            dense_array = np.array(bow_features)  # type: ignore
            
        df_bow = pd.DataFrame(dense_array,
                             columns=[f'bow_{i}' for i in range(bow_features.shape[1])])
        if 'type' in df.columns:
            df_bow['type'] = df['type'].values
    else:
        df_bow = bow_features  # type: ignore

    return df_bow, bow_vectorizer

def extract_tfidf_features(df, text_column='processed_text', max_features=6356):
    """
    Extract TF-IDF features

    Args:
        df (pd.DataFrame): DataFrame with text column
        text_column (str): Name of text column
        max_features (int): Maximum vocabulary size

    Returns:
        tuple: (df_tfidf, vectorizer)
    """
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_features = tfidf_vectorizer.fit_transform(df[text_column].astype(str).fillna(''))

    # Safe conversion of sparse matrix to dense
    if tfidf_features.shape[1] < 10000:
        # Check if it's a sparse matrix and convert safely
        if hasattr(tfidf_features, 'toarray'):
            dense_array = tfidf_features.toarray()  # type: ignore
        else:
            dense_array = np.array(tfidf_features)  # type: ignore
            
        df_tfidf = pd.DataFrame(dense_array,
                               columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        if 'type' in df.columns:
            df_tfidf['type'] = df['type'].values
    else:
        df_tfidf = tfidf_features  # type: ignore

    return df_tfidf, tfidf_vectorizer