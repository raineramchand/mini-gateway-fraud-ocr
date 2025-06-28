import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Load the trained XGBoost model
MODEL_PATH = Path("model.pkl")
model = joblib.load(MODEL_PATH)

# Pre-fitted label encoder from training
type_encoder = LabelEncoder()
type_encoder.classes_ = np.array(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])

def preprocess_transaction(tx: dict):
    """
    Convert raw transaction input into model-ready feature vector.
    """
    # Label encode the transaction type
    tx_type_encoded = type_encoder.transform([tx['type']])[0]

    features = [
        tx['step'],
        tx['amount'],
        tx['oldbalanceOrg'],
        tx['newbalanceOrig'],
        tx['oldbalanceDest'],
        tx['newbalanceDest'],
        tx['oldbalanceOrg'] - tx['newbalanceOrig'],   # balanceDiffOrig
        tx['newbalanceDest'] - tx['oldbalanceDest'],  # balanceDiffDest
        tx_type_encoded,
        tx['isFlaggedFraud']
    ]
    return np.array(features).reshape(1, -1)

def predict_transaction(tx: dict):
    """
    Predict fraud for a given transaction dictionary.
    Returns:
        - 0 or 1 (fraud label)
        - probability of fraud
    """
    features = preprocess_transaction(tx)
    label = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    return label, prob
