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

Local Setup and Installation
To run this project on your local machine, follow these steps:

1. Clone the repository:

Bash

git clone https://github.com/arpitmishra447/suicidal_message_classifier.git
cd suicidal_message_classifier
2. Create and activate a virtual environment:

Bash

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
3. Install dependencies:
The project uses Git LFS for large model files. Make sure you have it installed.

Bash

# Install Git LFS (if you haven't already)
git lfs install

# Install Python packages
pip install -r requirements.txt
4. Download Large Files:
Pull the large model files tracked by Git LFS.

Bash

git lfs pull
5. Run the Flask Application:

Bash

python app.py
The application will be available at http://127.0.0.1:8080.

How to Use the API
The deployed application has a /predict endpoint that accepts POST requests with a JSON body.

Endpoint: https://suicidal-message-classifier.onrender.com/predict

Method: POST

Request Body:

JSON

{
    "text": "Your text to be analyzed goes here."
}
Example using cURL:

Bash

curl -X POST \
  https://suicidal-message-classifier.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "I feel so empty and alone, I don'\''t think I can go on anymore."
  }'
Success Response (200 OK):

JSON

{
  "label": "suicide",
  "confidence": {
    "non-suicide": 0.08,
    "suicide": 0.92
  }
}
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