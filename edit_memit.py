import argparse
import time
from pathlib import Path

from src.real_easyedit import DATA_DIR, RESULT_DIR, load_json, runtime, save_json, setup_paths, summarize_easyedit, write_log


DEFAULT_HF_DATASET = "azhx/counterfact"


def parse_args():
    parser = argparse.ArgumentParser(description="Run MEMIT on real CounterFact records.")
    parser.add_argument("--cases", type=int, default=500, help="Number of CounterFact edit facts.")
    parser.add_argument("--batch-size", type=int, default=500, help="500 means one-shot MEMIT; smaller values run sequential chunks.")
    parser.add_argument("--mom2-samples", type=int, default=100000, help="Number of samples for covariance statistics.")
    parser.add_argument("--layers", default="4,5,6,7", help="Comma-separated MEMIT layers.")
    parser.add_argument("--data-path", default=None, help="Optional local CounterFact JSON file.")
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET, help="Hugging Face CounterFact dataset name.")
    parser.add_argument("--prepare-only", action="store_true", help="Only convert and save the 500-record MEMIT dataset.")
    return parser.parse_args()


def load_counterfact_records(args) -> list[dict]:
    if args.data_path:
        data_path = Path(args.data_path)
        records = load_json(data_path)
        return records[: args.cases]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load CounterFact. "
            "Install requirements.txt or pass --data-path to a local CounterFact JSON file."
        ) from exc

    dataset = load_dataset(args.hf_dataset, split=f"train[:{args.cases}]")
    return [dict(row) for row in dataset]


def pick_first(items, fallback=None):
    if isinstance(items, list) and items:
        return items[0]
    return fallback


def convert_counterfact_record(record: dict, idx: int) -> dict:
    rewrite = record["requested_rewrite"]
    subject = rewrite["subject"]
    prompt_template = rewrite["prompt"]
    prompt = prompt_template.format(subject) if "{}" in prompt_template else prompt_template
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite["target_true"]["str"]
    rephrase_prompt = pick_first(record.get("paraphrase_prompts"), prompt)
    locality_prompt = pick_first(record.get("neighborhood_prompts"), prompt)

    return {
        "case_id": record.get("case_id", idx),
        "source": "CounterFact",
        "subject": subject,
        "prompt": prompt,
        "target_new": target_new,
        "ground_truth": target_true,
        "rephrase_prompt": rephrase_prompt,
        "locality_prompt": locality_prompt,
        # CounterFact neighborhood prompts do not provide per-prompt answers.
        # EasyEdit's CounterFact examples use target_true as the locality answer.
        "locality_ground_truth": target_true,
    }


def load_memit_facts(args) -> list[dict]:
    records = load_counterfact_records(args)
    facts = [convert_counterfact_record(record, idx) for idx, record in enumerate(records)]
    if len(facts) != args.cases:
        raise ValueError(f"Expected {args.cases} records, got {len(facts)}.")
    return facts


def main() -> None:
    args = parse_args()
    setup_paths()

    facts = load_memit_facts(args)
    data_name = f"counterfact_{args.cases}_real_sample.json"
    save_json(DATA_DIR / data_name, facts)
    if args.prepare_only:
        write_log(
            "edit_memit_prepare",
            [
                "Task 3 MEMIT Dataset Preparation",
                f"source: CounterFact ({args.hf_dataset if not args.data_path else args.data_path})",
                f"cases: {len(facts)}",
                f"saved: data/{data_name}",
            ],
        )
        return

    from easyeditor import BaseEditor, MEMITHyperParams

    hparams = MEMITHyperParams.from_hparams("hparams_memit_qwen2_0_5b.yaml")
    hparams.layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    hparams.batch_size = args.batch_size
    hparams.mom2_n_samples = args.mom2_samples

    start = time.perf_counter()
    editor = BaseEditor.from_hparams(hparams)
    locality = {
        "neighborhood": {
            "prompt": [fact["locality_prompt"] for fact in facts],
            "ground_truth": [fact["locality_ground_truth"] for fact in facts],
        }
    }

    metrics, _, _ = editor.batch_edit(
        prompts=[fact["prompt"] for fact in facts],
        target_new=[fact["target_new"] for fact in facts],
        ground_truth=[fact["ground_truth"] for fact in facts],
        rephrase_prompts=[fact["rephrase_prompt"] for fact in facts],
        locality_inputs=locality,
        subject=[fact["subject"] for fact in facts],
        sequential_edit=args.batch_size < args.cases,
        verbose=False,
    )

    stats = runtime(start)
    summary = summarize_easyedit(metrics)
    out_name = f"memit_counterfact_{args.cases}_bs{args.batch_size}_mom2{args.mom2_samples}_results.json"
    save_json(RESULT_DIR / out_name, {"metrics_raw": metrics, "metrics": summary, "runtime": stats})
    write_log(
        "edit_memit_standard",
        [
            "Task 3 MEMIT Standard Batch Editing (EasyEdit + Qwen2.5-0.5B)",
            "dataset: CounterFact real records",
            f"data: data/{data_name}",
            f"cases: {summary['cases']}",
            f"batch_size: {args.batch_size}",
            f"layers: {hparams.layers}",
            f"mom2_adjustment: {hparams.mom2_adjustment}",
            f"mom2_samples: {hparams.mom2_n_samples}",
            f"ES: {summary['efficacy'] * 100:.1f}%",
            f"PS: {summary['generalization'] * 100:.1f}%",
            f"NS: {summary['locality'] * 100:.1f}%",
            f"elapsed seconds: {stats['elapsed_seconds']}",
            f"peak memory MB: {stats['peak_memory_mb']}",
            f"saved: artifacts/results/{out_name}",
        ],
    )


if __name__ == "__main__":
    main()
