Suicidal Message Classifier
This project is a machine learning application designed to classify text as either indicative of suicidal ideation or not. It uses a sophisticated NLP pipeline to process text and an XGBoost model to make predictions. The model is deployed as a web API using Flask on Render.

Live Application: https://suicidal-message-classifier.onrender.com

Features
Advanced Text Preprocessing: A robust pipeline that cleans text by normalizing Unicode, expanding contractions, converting emojis to text, and removing URLs and punctuation.

Hybrid Feature Engineering: Combines two types of features for a rich understanding of the text:

Semantic Vectors: Uses pre-trained GloVe embeddings to capture the meaning of words.

Engineered Features: Extracts specific, domain-relevant features like crisis keyword counts, negation counts, and word counts.

High-Performance Model: Employs a tuned XGBoost classifier, which is a powerful gradient boosting algorithm known for its accuracy and efficiency.

REST API: The trained model is exposed via a Flask API that accepts JSON requests and returns predictions.

Interactive Web UI: A simple, clean frontend allows users to interact with the model directly from their browser.

Cloud Deployment: The entire application is deployed on Render for public access and use.

Model Performance
The final XGBoost model was tuned using RandomizedSearchCV and evaluated on a held-out test set.

Accuracy: 90.85%

F1-Score: 0.91 (for both classes)

This indicates a strong, well-balanced model that is effective at identifying both suicidal and non-suicidal text.


Project Structure
/suicidal_message_classifier/
|-- artifacts/              # Saved models, scalers, and vocab files
|-- templates/
|   |-- index.html          # Frontend UI
|-- app.py                  # Main Flask application
|-- model.py                # Class to load models and run predictions
|-- preprocess.py           # Text cleaning and feature engineering functions
|-- requirements.txt        # Python dependencies
|-- render.yaml             # Configuration for Render deployment
|-- .gitattributes          # Config for Git LFS
|-- .gitignore