from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from flask import Flask, request, jsonify

application = Flask(__name__)

#curl -X POST "http://127.0.0.1:5000/load-model" -H "Content-Type: application/json" -d "{\"news\": \"This is fake news\"}"
@application.route('/load-model', methods=['POST', 'GET'])
def load_model():

    # Decode the JSON data from the request
    data = request.get_json()
    
    # Check if data was received correctly
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Now you can access the data as a dictionary
    news_value = data.get('news')

    if news_value is None:
        return jsonify({"error": "'news' field is required"}), 400
    
    if not isinstance(news_value, str):
        return jsonify({"error": "'news' must be a string"}), 400

    # Model loading
    loaded_model = None
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)

    vectorizer = None
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

    # How to use model to predict
    prediction = loaded_model.predict(vectorizer.transform([news_value]))[0]

    # Output will be 'FAKE' if fake, 'REAL' if real
    return prediction


@application.route('/')
def index():
    return load_model()


if __name__ == '__main__':
    application.run(debug=True)