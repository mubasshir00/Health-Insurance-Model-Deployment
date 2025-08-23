from typing import Any, Dict, List, Optional
from pathlib import Path
import json, joblib
import numpy as np


# --- JSON-safe converter ---


def _to_py(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    return x


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Trained_Model"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"


class Model:
    def __init__(self) -> None:
        self.pipeline = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.features: List[str] = meta["features"]
        self.has_proba = hasattr(self.pipeline, "predict_proba")


    def _row(self, inputs: Dict[str, Any]) -> List[float]:
        row: List[float] = []
        for k in self.features:
            if k not in inputs:
                raise ValueError(f"Missing required feature(s): [{k}]")
            v = inputs[k]
            if isinstance(v, str):
                v = float(v) if v.strip() != "" else 0.0
            elif isinstance(v, bool):
                v = 1.0 if v else 0.0
            else:
                v = _to_py(v)
            row.append(float(v))
        return row


    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        X = [self._row(inputs)]
        y = self.pipeline.predict(X)[0]
        out: Dict[str, Any] = {"label": _to_py(int(y) if isinstance(y, (bool, int, np.integer)) else y)}
        if self.has_proba:
            proba = _to_py(self.pipeline.predict_proba(X)[0])
            out["fraud_probability"] = float(proba[1]) if len(proba) > 1 else float(proba[0])
        return _to_py(out)


_model: Optional[Model] = None


def get_model() -> Model:
    global _model
    if _model is None:
        _model = Model()
    return _model