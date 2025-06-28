# Mini-Gateway Fraud & OCR Prototype

## 📋 Overview

AajPay’s pilot micro-merchant onboarding flow needs a fast, accurate MVP that:

* **Detects likely fraudulent sign-ups** in real-time using transaction snapshots
* **Extracts merchant name & total amount** from the first receipt image via OCR
* Exposes a **single FastAPI** endpoint for inference
* Is fully **Dockerized** for easy deployment

## 🚀 Features

1. **Fraud Detection**

   * XGBoost-based model trained on <5 MB data (imbalance ≈ 1 : 800)
   * Achieves ≥ 0.75 PR-AUC on held-out data
   * Saved as `model.pkl`

2. **OCR Pipeline**

   * Uses OpenCV to detect & deskew printed area
   * Runs EasyOCR and parses `merchant_name` & `total_amount`
   * Outputs `ocr_results.json` for audit

3. **REST Inference Service**

   * **Endpoint:** `POST /score`
   * **Request payload** (JSON):

     ```json
     {
       "transaction": {
         "step": 1,
         "type": "TRANSFER",
         "amount": 500.0,
         "oldbalanceOrg": 1500.0,
         "newbalanceOrig": 1000.0,
         "oldbalanceDest": 200.0,
         "newbalanceDest": 700.0,
         "isFlaggedFraud": 0,
         "device_id": "device_abc",                // optional
         "geo": { "lat": 40.7128, "lon": -74.0060 }, // optional
         "BIN": "123456"                           // optional
       },
       "receipt_path": "images/0.jpg"
     }
     ```
   * **Response** (JSON):

     ```json
     {
       "fraud_score": 0.05,
       "merchant_name": "Walmart",
       "total": 123.45
     }
     ```
   * **Performance:** p99 latency < 60 ms on CPU

## 📥 Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/mini-gateway-fraud-ocr.git
   cd mini-gateway-fraud-ocr
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally**

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Open Swagger UI**
   Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to explore and test the API interactively.

5. **Try an example**
   In Swagger UI, expand `POST /score`, click **Try it out**, paste the payload from above, then **Execute**.

   Or use `curl`:

   ```bash
   curl -X POST http://localhost:8000/score \
     -H "Content-Type: application/json" \
     -d '{
       "transaction": {
         "step": 1,
         "type": "TRANSFER",
         "amount": 500.0,
         "oldbalanceOrg": 1500.0,
         "newbalanceOrig": 1000.0,
         "oldbalanceDest": 200.0,
         "newbalanceDest": 700.0,
         "isFlaggedFraud": 0,
         "device_id": "device_abc",
         "geo": { "lat": 40.7128, "lon": -74.0060 },
         "BIN": "123456"
       },
       "receipt_path": "images/0.jpg"
     }'
   ```

   **Example JSON response**:

   ```json
   {
     "fraud_score": 0.05,
     "merchant_name": "Walmart",
     "total": 123.45
   }
   ```

## 🐳 Docker Usage

1. **Build the image**

   ```bash
   docker build -t mini-gateway-ocr .
   ```

2. **Run the container**

   ```bash
   docker run -d -p 8000:8000 mini-gateway-ocr
   ```

3. **Invoke**

   ```bash
   curl -X POST http://localhost:8000/score \
     -H 'Content-Type: application/json' \
     -d '{
       "transaction": {
         "step": 1,
         "type": "TRANSFER",
         "amount": 500.0,
         "oldbalanceOrg": 1500.0,
         "newbalanceOrig": 1000.0,
         "oldbalanceDest": 200.0,
         "newbalanceDest": 700.0,
         "isFlaggedFraud": 0,
         "device_id": "device_abc",
         "geo": { "lat": 40.7128, "lon": -74.0060 },
         "BIN": "123456"
       },
       "receipt_path": "images/1.jpg"
     }'
   ```

## 📂 Project Structure

```
mini-gateway-fraud-ocr/
├── app.py                # FastAPI server
├── model_utils.py        # Train & load XGBoost model
├── ocr_module.py         # Receipt detection & OCR parsing
├── model.pkl             # Saved XGBoost model
├── requirements.txt      # Pinned dependencies
├── Dockerfile            # Container build instructions
├── .dockerignore         # Files to exclude from build
├── notebooks/            # EDA & baseline modeling
├── tests/                # Unit & integration tests
└── README.md             # (this file)
```

