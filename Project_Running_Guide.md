# ML Inference Service


FastAPI service that serves a trained RandomForest model.


## Run locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000