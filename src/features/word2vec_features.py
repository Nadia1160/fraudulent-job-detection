"""Word2Vec feature extraction"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
from typing import Optional, List, Any

class Word2VecExtractor:
    """Word2Vec feature extractor"""
    
    def __init__(self, vector_size=100, window=5, min_count=2, workers=2, epochs=10):
        """
        Initialize Word2Vec extractor
        
        Args:
            vector_size (int): Dimension of word vectors
            window (int): Context window size
            min_count (int): Minimum word frequency
            workers (int): Number of worker threads
            epochs (int): Number of training epochs
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model: Optional[Word2Vec] = None

    def tokenize_texts(self, texts):
        """Tokenize texts for Word2Vec training"""
        tokenized = []
        for text in tqdm(texts, desc="Tokenizing", unit="docs"):
            if isinstance(text, str):
                tokens = [word.lower() for word in text.split() if len(word) > 2]
                tokenized.append(tokens)
            else:
                tokenized.append([])
        return tokenized

    def fit(self, texts):
        """Train Word2Vec model"""
        sentences = self.tokenize_texts(texts)
        sentences = [s for s in sentences if len(s) > 0]

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1,
            epochs=self.epochs,
            seed=42
        )
        return self

    def transform(self, texts):
        """Transform texts to document vectors"""
        # Check if model is fitted
        if self.model is None:
            raise ValueError("Model must be fitted first. Call fit() or fit_transform().")
        
        sentences = self.tokenize_texts(texts)

        def get_document_vector(tokens):
            vectors = []
            # Now we know self.model is not None, but Pylance doesn't
            for token in tokens:
                if token in self.model.wv:  # type: ignore
                    vectors.append(self.model.wv[token])  # type: ignore

            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(self.vector_size)

        doc_vectors = []
        for tokens in tqdm(sentences, desc="Creating vectors", unit="docs"):
            doc_vectors.append(get_document_vector(tokens))

        return np.array(doc_vectors)

    def fit_transform(self, texts):
        """Fit model and transform texts"""
        self.fit(texts)
        return self.transform(texts)

def extract_word2vec_features(df, text_column='processed_text', vector_size=100):
    """
    Extract Word2Vec features
    
    Args:
        df (pd.DataFrame): DataFrame with text column
        text_column (str): Name of text column
        vector_size (int): Dimension of word vectors
        
    Returns:
        tuple: (df_w2v, model)
    """
    extractor = Word2VecExtractor(vector_size=vector_size)
    doc_vectors = extractor.fit_transform(df[text_column].tolist())

    df_w2v = pd.DataFrame(doc_vectors, columns=[f'w2v_{i}' for i in range(vector_size)])
    if 'type' in df.columns:
        df_w2v['type'] = df['type'].iloc[:len(doc_vectors)].values

    return df_w2v, extractor.model