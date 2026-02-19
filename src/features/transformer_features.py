"""Transformer-based feature extraction"""

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class CPUFriendlyTransformerFeatures:
    """CPU-friendly transformer feature extractor"""

    def __init__(self, model_path, model_name, max_length=128):
        """
        Initialize transformer feature extractor

        Args:
            model_path (str): Path to transformer model
            model_name (str): Name of the model
            max_length (int): Maximum sequence length
        """
        self.model_path = model_path
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cpu')

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
            self.model.eval()
        except Exception as e:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                ).to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Error loading model: {e}")
                raise e

    def extract_features(self, texts):
        """
        Extract features from texts

        Args:
            texts (list): List of text strings

        Returns:
            np.array: Feature matrix
        """
        features = []
        batch_size = 4

        for i in tqdm(range(0, len(texts), batch_size), desc=f"   {self.model_name}", unit="batch"):
            batch_texts = texts[i:i+batch_size]

            try:
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                features.extend(batch_features)

            except Exception as e:
                # Add zero vectors for failed batch
                features.extend([np.zeros(384)] * len(batch_texts))

        return np.array(features)