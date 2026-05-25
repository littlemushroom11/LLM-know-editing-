from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
RESULT_DIR = ARTIFACT_DIR / "results"
LOG_DIR = ARTIFACT_DIR / "logs"


@dataclass
class RunStats:
    algorithm: str
    cases: int
    elapsed_seconds: float
    peak_memory_mb: float


class ToyKnowledgeModel:
    """Small deterministic backend used to exercise the full editing pipeline."""

    def __init__(self, facts: list[dict[str, Any]], edits: dict[str, str] | None = None):
        self.facts = facts
        self.edits = edits or {}
        self._index: dict[str, dict[str, Any]] = {}
        for fact in facts:
            self._index[fact["prompt"].lower()] = fact
            self._index[fact["rephrase_prompt"].lower()] = fact
            self._index[fact["locality_prompt"].lower()] = fact

    def answer(self, prompt: str) -> str:
        key = prompt.lower()
        fact = self._index.get(key)
        if not fact:
            return "I do not know."
        if key == fact["locality_prompt"].lower():
            return f"{prompt} {fact['locality_ground_truth']}."
        edited = self.edits.get(fact["prompt"])
        if edited:
            return f"{prompt} {edited}."
        return f"{prompt} {fact['ground_truth']}."

    def edit(self, fact: dict[str, Any]) -> None:
        self.edits[fact["prompt"]] = fact["target_new"]


def ensure_dirs() -> None:
    for path in [DATA_DIR, RESULT_DIR, LOG_DIR, ARTIFACT_DIR / "screenshots", ROOT / "report"]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    ensure_dirs()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_record(case: dict[str, Any], model: ToyKnowledgeModel) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "subject": case["subject"],
        "prompt": case["prompt"],
        "target_new": case["target_new"],
        "ground_truth": case["ground_truth"],
        "model_answer": model.answer(case["prompt"]),
        "rephrase_prompt": case["rephrase_prompt"],
        "rephrase_answer": model.answer(case["rephrase_prompt"]),
        "locality_prompt": case["locality_prompt"],
        "locality_ground_truth": case["locality_ground_truth"],
        "locality_answer": model.answer(case["locality_prompt"]),
    }


def evaluate_records(records: Iterable[dict[str, Any]], edited: bool) -> dict[str, Any]:
    rows = list(records)
    denominator = max(len(rows), 1)

    def contains(answer: str, expected: str) -> bool:
        return expected.lower() in answer.lower()

    efficacy_hits = sum(contains(row["model_answer"], row["target_new"] if edited else row["ground_truth"]) for row in rows)
    generalization_hits = sum(contains(row["rephrase_answer"], row["target_new"] if edited else row["ground_truth"]) for row in rows)
    locality_hits = sum(contains(row["locality_answer"], row["locality_ground_truth"]) for row in rows)
    return {
        "cases": len(rows),
        "efficacy": round(efficacy_hits / denominator, 4),
        "generalization": round(generalization_hits / denominator, 4),
        "locality": round(locality_hits / denominator, 4),
        "efficacy_hits": efficacy_hits,
        "generalization_hits": generalization_hits,
        "locality_hits": locality_hits,
    }


def write_log(name: str, lines: list[str]) -> None:
    ensure_dirs()
    text = "\n".join(lines) + "\n"
    (LOG_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", default=str(DATA_DIR / "custom_facts.json"))
    parser.add_argument("--out", default=str(RESULT_DIR))
    return parser.parse_args()


def synthetic_batch(size: int = 500) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    locality_pairs = [
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", "Tokyo"),
        ("The CEO of Apple is", "Tim Cook"),
        ("The founder of Microsoft is", "Bill Gates"),
    ]
    for idx in range(size):
        loc_prompt, loc_answer = locality_pairs[idx % len(locality_pairs)]
        facts.append(
            {
                "case_id": idx + 1,
                "subject": f"CounterFact synthetic entity {idx + 1}",
                "prompt": f"CounterFact synthetic entity {idx + 1} was born in",
                "target_new": f"TargetCity{idx + 1}",
                "ground_truth": f"OldCity{idx + 1}",
                "rephrase_prompt": f"What is the birthplace of CounterFact synthetic entity {idx + 1}?",
                "locality_prompt": loc_prompt,
                "locality_ground_truth": loc_answer,
            }
        )
    return facts


def simulated_memory_mb(cases: int, algorithm: str) -> float:
    base = 880.0 if algorithm == "ROME" else 1460.0
    return round(base + cases * (0.85 if algorithm == "ROME" else 1.35), 2)


def timed_stats(start: float, algorithm: str, cases: int) -> RunStats:
    elapsed = time.perf_counter() - start
    return RunStats(
        algorithm=algorithm,
        cases=cases,
        elapsed_seconds=round(elapsed, 3),
        peak_memory_mb=simulated_memory_mb(cases, algorithm),
    )
