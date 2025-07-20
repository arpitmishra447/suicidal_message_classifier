# save_glove.py
import gensim.downloader as api
from gensim.models import KeyedVectors

print("Downloading GloVe model...")
glove_model = api.load('glove-wiki-gigaword-100')

# Save in an efficient, load-optimized format
save_path = "artifacts/glove-wiki-gigaword-100.kv"
glove_model.save(save_path)

print(f"GloVe model downloaded and saved to {save_path}")