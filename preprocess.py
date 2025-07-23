import re
import emoji
import contractions
import unicodedata
import numpy as np
import pandas as pd

# Your functions for cleaning and feature engineering
def remove_unusual_unicode_but_keep_emojis(text):
    return ''.join(c for c in text if ord(c) <= 0xFFFF or emoji.is_emoji(c))
def safe_expand_contractions(text):
    try:
        text = text.replace("İ", "I")
        text = remove_unusual_unicode_but_keep_emojis(text)
        return contractions.fix(text)
    except Exception as e:
        print(f"[EXPANSION ERROR] {e} | text: {text[:100]}")
        return text

def clean_text(text, keep_emojis=True):
    try:
        # Normalize Unicode (cleans up 'İ' or curly quotes)
        text = unicodedata.normalize("NFKC", text)

        # Expand contractions safely
        text = safe_expand_contractions(text)

        # Lowercase after expansion (to preserve casing rules first)
        text = text.lower()

        # Emoji handling
        if keep_emojis:
            text = emoji.demojize(text, delimiters=(" ", " "))
        else:
            text = emoji.replace_emoji(text, "")

        # Replace URLs with token
        text = re.sub(r"http\S+|www\S+|https\S+", "<url>", text)

        # Add space around punctuation
        text = re.sub(r"([\"\'\.\(\)\!\?\-\\\/\,])", r" \1 ", text)

        # Normalize whitespace
        return " ".join(text.split())

    except Exception as e:
        print(f"[CLEAN_TEXT ERROR] {e} | text: {text[:100]}")
        return "<MALFORMED>"

class FeatureEngineer:
    def __init__(self):
        # Precompiled regex patterns
        self.crisis_pat = re.compile(
            r"(i want to die|i want to kill myself|i am going to end my life|i have nothing left|"
            r"goodbye forever|no reason to live|i cannot do this anymore|ending it all|"
            r"final goodbye|life is meaningless|i am done living|i do not want to live anymore)",
            flags=re.IGNORECASE
        )
        self.self_ref_pat = re.compile(r'\b(i|me|my|myself)\b', flags=re.IGNORECASE)
        self.negation_pat = re.compile(r'\b(no|not|never|nothing)\b', flags=re.IGNORECASE)

    def extract_features(self, series: pd.Series) -> np.ndarray:
        """Vectorized feature extraction using pandas string methods"""
        # Lowercase for consistency
        series = series.astype(str).str.lower()

        features = pd.DataFrame()
        features['crisis_count'] = series.str.count(self.crisis_pat)
        features['self_ref'] = series.str.count(self.self_ref_pat)
        features['negations'] = series.str.count(self.negation_pat)
        features['char_count'] = series.str.len()
        features['word_count'] = series.str.split().str.len()
        features['exclamation_count'] = series.str.count(r'!')
        features['question_count'] = series.str.count(r'\?')
        features['ellipsis_count'] = series.str.count(r'\.\.\.')

        return features.to_numpy(dtype=np.float32)
    
