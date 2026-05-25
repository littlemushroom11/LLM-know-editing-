import time

import torch

from src.real_easyedit import (
    DATA_DIR,
    RESULT_DIR,
    contains,
    load_json,
    runtime,
    save_json,
    setup_paths,
    summarize_easyedit,
    write_log,
)


def generate_answer(model, tok, prompt: str, max_new_tokens: int = 32) -> str:
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
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


def restore_weights(model, weights_copy) -> None:
    from easyeditor.util import nethook

    if callable(weights_copy):
        weights_copy()
        return

    with torch.no_grad():
        for name, value in weights_copy.items():
            parameter = nethook.get_parameter(model, name)
            parameter[...] = value.to(parameter.device)


def generate_after_rome_edit(editor, request: dict, fact: dict) -> dict:
    edited_model, weights_copy = editor.apply_algo(
        editor.model,
        editor.tok,
        [request],
        editor.hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
    )
    try:
        direct_answer = generate_answer(edited_model, editor.tok, fact["prompt"])
        rephrase_answer = generate_answer(edited_model, editor.tok, fact["rephrase_prompt"])
        locality_answer = generate_answer(edited_model, editor.tok, fact["locality_prompt"])
        return {
            **fact,
            "edited_model_answer": direct_answer,
            "edited_rephrase_answer": rephrase_answer,
            "edited_locality_answer": locality_answer,
            "target_hit": contains(direct_answer, fact["target_new"]),
            "rephrase_target_hit": contains(rephrase_answer, fact["target_new"]),
            "locality_hit": contains(locality_answer, fact["locality_ground_truth"]),
        }
    finally:
        restore_weights(editor.model, weights_copy)


def main() -> None:
    setup_paths()
    from easyeditor import BaseEditor, ROMEHyperParams

    facts = load_json(DATA_DIR / "custom_facts.json")
    hparams = ROMEHyperParams.from_hparams("hparams_rome_qwen2_0_5b.yaml")
    start = time.perf_counter()
    editor = BaseEditor.from_hparams(hparams)
    metrics = []
    generation_records = []
    for fact in facts:
        locality = {
            "neighborhood": {
                "prompt": [fact["locality_prompt"]],
                "ground_truth": [fact["locality_ground_truth"]],
            }
        }
        cur_metrics, _, _ = editor.edit(
            prompts=[fact["prompt"]],
            target_new=[fact["target_new"]],
            ground_truth=[fact["ground_truth"]],
            rephrase_prompts=[fact["rephrase_prompt"]],
            locality_inputs=locality,
            subject=[fact["subject"]],
            verbose=False,
        )
        metrics.extend(cur_metrics)
        generation_records.append(generate_after_rome_edit(editor, cur_metrics[0]["requested_rewrite"], fact))
    stats = runtime(start)
    summary = summarize_easyedit(metrics)
    save_json(
        RESULT_DIR / "rome_results.json",
        {
            "metrics_raw": metrics,
            "generation_records": generation_records,
            "metrics": summary,
            "runtime": stats,
        },
    )
    write_log(
        "edit_rome",
        [
            "Task 2 ROME Single Fact Editing (EasyEdit + Qwen2.5-0.5B)",
            f"cases: {summary['cases']}",
            f"pre ES: {summary['pre_efficacy'] * 100:.1f}%",
            f"ES: {summary['efficacy'] * 100:.1f}%",
            f"PS: {summary['generalization'] * 100:.1f}%",
            f"NS: {summary['locality'] * 100:.1f}%",
            f"elapsed seconds: {stats['elapsed_seconds']}",
            f"peak memory MB: {stats['peak_memory_mb']}",
            "model answers saved: generation_records in artifacts/results/rome_results.json",
            "saved: artifacts/results/rome_results.json",
        ],
    )


if __name__ == "__main__":
    main()
