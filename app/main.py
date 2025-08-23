import time
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from .schemas import PredictRequest, PredictResponse, PredictAsyncRequest, Ack
from .model import get_model
from .security import sign_payload, HEADER_NAME

app = FastAPI(title="ML Inference Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    t0 = time.perf_counter_ns()
    result = get_model().predict(req.inputs)
    latency_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
    return PredictResponse(id=req.id, result=result, latency_ms=latency_ms)

async def _send_callback(callback_url: str, payload: dict):
    # Lazy import so httpx isn’t required unless async path used
    import httpx
    headers = {HEADER_NAME: sign_payload(payload)}
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(callback_url, json=payload, headers=headers)

@app.post("/predict-async", response_model=Ack, status_code=202)
async def predict_async(req: PredictAsyncRequest, bg: BackgroundTasks):
    result = get_model().predict(req.inputs)
    payload = {"id": req.id, "result": result}
    bg.add_task(_send_callback, req.callback_url, payload)
    return Ack(status="accepted", id=req.id)

@app.exception_handler(Exception)
async def handle_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})
