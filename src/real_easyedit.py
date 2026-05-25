from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "artifacts" / "results"
LOG_DIR = ROOT / "artifacts" / "logs"
DATA_DIR = ROOT / "data"
MODEL_PATH = "D:/hf_models/Qwen2.5-0.5B-Instruct"
EASYEDIT_PATH = "D:/EasyEdit-main"
DEVICE = "cuda:0"


def setup_paths() -> None:
    if EASYEDIT_PATH not in sys.path:
        sys.path.insert(0, EASYEDIT_PATH)


def ensure_dirs() -> None:
    for path in [RESULT_DIR, LOG_DIR, DATA_DIR, ROOT / "artifacts" / "screenshots", ROOT / "report"]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> None:
    ensure_dirs()
    Path(path).write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list) or isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_log(name: str, lines: list[str]) -> None:
    ensure_dirs()
    text = "\n".join(lines) + "\n"
    (LOG_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def load_hf_model(dtype: torch.dtype = torch.float16):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    return model, tok


def generate(model, tok, prompt: str, max_new_tokens: int = 16) -> str:
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    return tok.decode(output[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip()


def contains(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def metric_mean(metrics: list[dict[str, Any]], section: str, key: str, default: float = 0.0) -> float:
    values: list[float] = []
    for item in metrics:
        cur = item.get(section, {}).get(key)
        if isinstance(cur, list):
            values.extend(float(x) for x in cur)
        elif cur is not None:
            values.append(float(cur))
    return sum(values) / len(values) if values else default


def locality_mean(metrics: list[dict[str, Any]], default: float = 0.0) -> float:
    values: list[float] = []
    for item in metrics:
        loc = item.get("post", {}).get("locality", {})
        for key, cur in loc.items():
            if key.endswith("_acc"):
                if isinstance(cur, list):
                    values.extend(float(x) for x in cur)
                else:
                    values.append(float(cur))
    return sum(values) / len(values) if values else default


def summarize_easyedit(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(metrics),
        "efficacy": round(metric_mean(metrics, "post", "rewrite_acc"), 4),
        "generalization": round(metric_mean(metrics, "post", "rephrase_acc"), 4),
        "locality": round(locality_mean(metrics), 4),
        "pre_efficacy": round(metric_mean(metrics, "pre", "rewrite_acc"), 4),
        "pre_generalization": round(metric_mean(metrics, "pre", "rephrase_acc"), 4),
    }


def runtime(start: float) -> dict[str, float]:
    return {
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        if torch.cuda.is_available()
        else 0.0,
    }
