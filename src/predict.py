#Function to get encoded and scaled data from preprocessing.py and then predicting according to it
from src.preprocessing import preprocess_input
import joblib
model = joblib.load("models/best_model.pkl")
encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")


def predict_price(input_data,):
    processed_input = preprocess_input(input_data,encoder,scaler)
    prediction = model.predict(processed_input)
    return prediction   