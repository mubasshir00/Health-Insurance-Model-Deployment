import time
from typing import Any, Dict

class Model:
    def __init__(self) -> None:
        time.sleep(0.05)  # simulate load

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = str(inputs.get("text", ""))
        score = min(0.999, 0.5 + len(text) * 0.01)
        label = "POSITIVE" if score >= 0.5 else "NEGATIVE"
        return {"score": round(score, 3), "label": label}

_model: Model | None = None

def get_model() -> Model:
    global _model
    if _model is None:
        _model = Model()
    return _model
