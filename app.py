from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('banglore_home_prices_model.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the JSON request
        data = request.get_json()
        
        # Convert JSON data to a DataFrame
        input_data = pd.DataFrame([data])

        # Perform predictions
        predictions = model.predict(input_data)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
