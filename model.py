# model.py
import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tokenizers import BertWordPieceTokenizer
from preprocess import clean_text, FeatureEngineer

class SuicideModel:
    def __init__(self, model_path, scaler_path, vocab_path, glove_path):
        print("Loading model and artifacts...")
        # Load the trained model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load the tokenizer and GloVe vectors
        self.tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=True)
        self.glove_model = KeyedVectors.load(glove_path, mmap='r')
        
        # Initialize the feature engineer
        self.feature_engineer = FeatureEngineer()
        print("Model and artifacts loaded successfully.")

    def _sentence_to_vector(self, text):
        tokens = self.tokenizer.encode(str(text)).tokens
        tokens = [word for word in tokens if word not in ['[CLS]', '[SEP]']]
        word_vectors = [self.glove_model[word] for word in tokens if word in self.glove_model]
        if not word_vectors:
            return np.zeros(self.glove_model.vector_size)
        return np.mean(word_vectors, axis=0)

    def predict(self, text_input):
        # 1. Clean the raw text
        cleaned_text = clean_text(text_input)
        
        # 2. Extract and scale engineered features
        # We wrap the text in a pandas Series to use the vectorized methods
        text_series = pd.Series([cleaned_text])
        engineered_features = self.feature_engineer.extract_features(text_series)
        scaled_features = self.scaler.transform(engineered_features)
        
        # 3. Create GloVe vector
        glove_vector = self._sentence_to_vector(cleaned_text).reshape(1, -1)
        
        # 4. Combine all features
        combined_features = np.hstack([glove_vector, scaled_features])
        
        # 5. Make prediction and get probabilities
        prediction = self.model.predict(combined_features)[0]
        probability = self.model.predict_proba(combined_features)[0].tolist()
        
        # Map integer prediction back to label
        label = "suicide" if prediction == 1 else "non-suicide"
        
        return {"label": label, "confidence": {"non-suicide": probability[0], "suicide": probability[1]}}