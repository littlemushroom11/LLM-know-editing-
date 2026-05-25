import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "artifacts" / "results"
LOG_DIR = ROOT / "artifacts" / "logs"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_log(name: str, lines: list[str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (LOG_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Baseline, ROME, and MEMIT evaluation metrics.")
    parser.add_argument(
        "--memit-result",
        default=None,
        help="Optional MEMIT result JSON. If omitted, the newest memit_counterfact_*_results.json is used.",
    )
    return parser.parse_args()


def resolve_memit_result(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
        return path if path.is_absolute() else RESULT_DIR / path

    counterfact_runs = sorted(
        RESULT_DIR.glob("memit_counterfact_*_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if counterfact_runs:
        return counterfact_runs[0]

    legacy_standard_runs = sorted(
        RESULT_DIR.glob("memit_standard_*_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if legacy_standard_runs:
        return legacy_standard_runs[0]

    return RESULT_DIR / "memit_results.json"


def load_metrics(path: Path, name: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{name} result not found: {path}")
    payload = load_json(path)
    if "metrics" not in payload:
        raise KeyError(f"{name} result has no 'metrics' field: {path}")
    return payload["metrics"]


def main() -> None:
    args = parse_args()
    memit_path = resolve_memit_result(args.memit_result)

    baseline = load_metrics(RESULT_DIR / "baseline_results.json", "Baseline")
    rome = load_metrics(RESULT_DIR / "rome_results.json", "ROME")
    memit = load_metrics(memit_path, "MEMIT")
    summary = {
        "Baseline": {
            "cases": baseline["cases"],
            "efficacy": baseline["target_hit_rate"],
            "generalization": 0.0,
            "locality": baseline["locality_generation_rate"],
        },
        "ROME": rome,
        "MEMIT": memit,
        "_sources": {
            "baseline": "baseline_results.json",
            "rome": "rome_results.json",
            "memit": memit_path.name,
        },
    }
    save_json(RESULT_DIR / "evaluation_summary.json", summary)
    lines = [
        "Task 4 Comprehensive Evaluation",
        f"MEMIT source: artifacts/results/{memit_path.name}",
        "Algorithm | Cases | ES | PS | NS",
    ]
    for name in ["Baseline", "ROME", "MEMIT"]:
        metrics = summary[name]
        lines.append(
            f"{name} | {metrics['cases']} | {metrics['efficacy'] * 100:.1f}% | "
            f"{metrics['generalization'] * 100:.1f}% | {metrics['locality'] * 100:.1f}%"
        )
    lines.append("saved: artifacts/results/evaluation_summary.json")
    write_log("evaluate", lines)


if __name__ == "__main__":
    main()
