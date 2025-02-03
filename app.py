from flask import Flask, request, jsonify
import joblib

# Load saved model and metadata
model_data = joblib.load('iris_model.pkl')
model = model_data['model']
target_names = model_data['target_names']

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get parameters from request
        params = {
            'sepal_length': float(request.args.get('sepal_length')),
            'sepal_width': float(request.args.get('sepal_width')),
            'petal_length': float(request.args.get('petal_length')),
            'petal_width': float(request.args.get('petal_width'))
        }
        
        # Create feature array
        features = [[params['sepal_length'], 
                   params['sepal_width'], 
                   params['petal_length'], 
                   params['petal_width']]]
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features).max()
        
        # Standardized response format
        response = {
            "model_type": "LogisticRegression",
            "prediction": int(prediction[0]),
            "species": target_names[prediction[0]],
            "confidence": float(probability),
            "parameters": params
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "Iris Species Prediction API - Use /predict endpoint with parameters"

if __name__ == '__main__':
    app.run()