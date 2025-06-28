import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import os

from ocr_module import extract_receipt_info
from model_utils import predict_transaction

app = FastAPI()

# Request schema matching features expected by model_utils
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int
    # Optional extra fields (not used by model_utils but kept for other purposes)
    device_id: Optional[str]
    geo: Optional[Dict[str, float]]
    BIN: Optional[str]

class ScoreRequest(BaseModel):
    transaction: Transaction
    receipt_path: str

# Response schema
class ScoreResponse(BaseModel):
    fraud_score: float
    merchant_name: str
    total: float

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    try:
        # Convert to dict and predict
        tx_dict = req.transaction.dict()
        _, fraud_prob = predict_transaction(tx_dict)

        # Check receipt path
        if not os.path.exists(req.receipt_path):
            raise FileNotFoundError(f"Receipt file not found: {req.receipt_path}")

        # Extract merchant info via OCR
        merchant_name, total, _ = extract_receipt_info(req.receipt_path)

        return ScoreResponse(
            fraud_score=float(fraud_prob),
            merchant_name=merchant_name,
            total=float(total),
        )

    except FileNotFoundError as e:
        # Client error if receipt not found
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
