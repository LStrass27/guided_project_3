from flask import Flask, request, jsonify
import pickle
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        # Drop the specified columns
        return X_transformed.drop(columns=self.columns_to_drop, errors='ignore')

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    if isinstance(data, dict):
        data = [data]

    prediction = model.predict(pd.DataFrame(data))

    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(port=5000)
