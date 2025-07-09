from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

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
