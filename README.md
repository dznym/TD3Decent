# Iris Species Prediction API

A machine learning system for iris species classification with a Flask API and model consensus evaluation.

## Features

- **Machine Learning Model**: Logistic Regression classifier trained on Iris dataset
- **REST API**: Flask-based prediction endpoint
- **Distributed Testing**: Uses ngrok tunnels to test multiple model endpoints
- **Model Metadata**: Includes feature names and target classes in model package
- **Consensus System**: Meta app for testing multiple model instances with credit-based scoring

## Installation

1. **Install requirements**
```bash
pip install -r requirements.txt
```

## Usage
1. **Train and Save Model**

```bash
python model.py
```

2. **Start Flask API**
```bash
python app.py
```

3. **Make Prediction Request**

Example using cURL:
```bash
curl "http://localhost:5000/predict?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2"
```
Example Response:
```json
{
  "model_type": "LogisticRegression",
  "prediction": 0,
  "species": "setosa",
  "confidence": 0.98,
  "parameters": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

4. **Run meta_app.py**

```bash
python meta_app.py
```
    Tests predefined samples against multiple model endpoints

    Implements credit system based on prediction consensus

    Outputs predictions and final model credits
  

   