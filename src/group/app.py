import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
from keras.models import load_model
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='model_predictions.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Load the trained model from .keras file
model = load_model('best_model.keras')

# Load other necessary objects
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    
    # Preprocess the text
    text_vectorized = tfidf_vectorizer.transform([text]).toarray()
    text_vectorized_reshaped = np.expand_dims(text_vectorized, axis=1)
    
    # Predict using the loaded model
    predictions = model.predict(text_vectorized_reshaped)
    predicted_label = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    # Log the prediction
    app.logger.info(f"Input: {text}, Prediction: {predicted_label[0]}")
    
    return jsonify({'prediction': predicted_label[0]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting app on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)
