"""Data preprocessing module"""

import re
import numpy as np
import pandas as pd
from sklearn.utils import resample

def preprocess_text(text):
    """
    Preprocess text as per paper's methodology
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s.,!?$%]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_dataset(df_annotated):
    """
    Prepare and balance dataset as per paper methodology
    
    Args:
        df_annotated (pd.DataFrame): Annotated dataset with 'type' column
        
    Returns:
        pd.DataFrame: Balanced and preprocessed dataset
    """
    # Isolate fraudulent jobs (type > 0)
    fraudulent_df = df_annotated[df_annotated['type'] > 0].copy()
    real_df = df_annotated[df_annotated['type'] == 0].copy()
    
    # Sample real jobs to match fraudulent count
    n_fraudulent = len(fraudulent_df)
    n_real_to_sample = min(n_fraudulent, len(real_df))
    real_sampled = real_df.sample(n=n_real_to_sample, random_state=42)
    
    # Combine - FIXED: Create explicit list and use type ignore
    dfs_to_concat = [fraudulent_df, real_sampled]
    df_balanced = pd.concat(dfs_to_concat, axis=0, ignore_index=True)  # type: ignore
    
    # Upsample Type 3 (MLM)
    type3_df = df_balanced[df_balanced['type'] == 3]
    
    if len(type3_df) > 0:
        target_type3 = max(150, len(type3_df) * 2)
        n_samples_needed = target_type3 - len(type3_df)
        
        if n_samples_needed > 0:
            type3_upsampled = resample(type3_df,
                                      replace=True,
                                      n_samples=n_samples_needed,
                                      random_state=42)
            # FIXED: Create explicit list and use type ignore
            dfs_to_concat = [df_balanced, type3_upsampled]
            df_balanced = pd.concat(dfs_to_concat, axis=0, ignore_index=True)  # type: ignore
    
    # Combine text fields
    df_balanced['combined_text'] = (
        df_balanced['company_profile'].fillna('') + ' ' +
        df_balanced['description'].fillna('') + ' ' +
        df_balanced['requirements'].fillna('') + ' ' +
        df_balanced['benefits'].fillna('')
    )
    
    # Apply preprocessing
    df_balanced['processed_text'] = df_balanced['combined_text'].apply(preprocess_text)
    
    return df_balanced