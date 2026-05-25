'''

import time

from src.real_easyedit import (
    DATA_DIR,
    RESULT_DIR,
    contains,
    load_hf_model,
    load_json,
    generate,
    runtime,
    save_json,
    write_log,
)


def main() -> None:
    start = time.perf_counter()
    facts = load_json(DATA_DIR / "custom_facts.json")
    model, tok = load_hf_model()
    records = []
    for fact in facts:
        answer = generate(model, tok, fact["prompt"])
        rephrase_answer = generate(model, tok, fact["rephrase_prompt"])
        locality_answer = generate(model, tok, fact["locality_prompt"])
        records.append(
            {
                **fact,
                "model_answer": answer,
                "rephrase_answer": rephrase_answer,
                "locality_answer": locality_answer,
                "target_hit": contains(answer, fact["target_new"]),
                "ground_truth_hit": contains(answer, fact["ground_truth"]),
                "locality_hit": contains(locality_answer, fact["locality_ground_truth"]),
            }
        )
    n = len(records)
    metrics = {
        "cases": n,
        "target_hit_rate": round(sum(r["target_hit"] for r in records) / n, 4),
        "old_knowledge_hit_rate": round(sum(r["ground_truth_hit"] for r in records) / n, 4),
        "locality_generation_rate": round(sum(r["locality_hit"] for r in records) / n, 4),
    }
    stats = runtime(start)
    save_json(RESULT_DIR / "baseline_results.json", {"records": records, "metrics": metrics, "runtime": stats})
    write_log(
        "baseline",
        [
            "Task 1 Baseline Evaluation (Qwen2.5-0.5B-Instruct)",
            f"cases: {metrics['cases']}",
            f"target answer hit before editing: {metrics['target_hit_rate'] * 100:.1f}%",
            f"old answer hit before editing: {metrics['old_knowledge_hit_rate'] * 100:.1f}%",
            f"locality generation sanity check: {metrics['locality_generation_rate'] * 100:.1f}%",
            f"elapsed seconds: {stats['elapsed_seconds']}",
            f"peak memory MB: {stats['peak_memory_mb']}",
            "saved: artifacts/results/baseline_results.json",
        ],
    )


if __name__ == "__main__":
    main()
'''

import time

from src.real_easyedit import (
    DATA_DIR,
    RESULT_DIR,
    contains,
    load_hf_model,
    load_json,
    generate,
    runtime,
    save_json,
    write_log,
)


def main() -> None:
    start = time.perf_counter()

    # 1. 读取自定义 10 条事实更新数据
    facts = load_json(DATA_DIR / "custom_facts.json")

    # 2. 加载未编辑前的原始模型
    model, tok = load_hf_model()

    records = []

    print("=" * 100)
    print("Task 1 Baseline Evaluation: 原始模型编辑前回答结果")
    print("=" * 100)

    # 3. 逐条测试模型回答
    for idx, fact in enumerate(facts, start=1):
        answer = generate(model, tok, fact["prompt"])
        rephrase_answer = generate(model, tok, fact["rephrase_prompt"])
        locality_answer = generate(model, tok, fact["locality_prompt"])

        target_hit = contains(answer, fact["target_new"])
        ground_truth_hit = contains(answer, fact["ground_truth"])
        locality_hit = contains(locality_answer, fact["locality_ground_truth"])

        record = {
            **fact,
            "model_answer": answer,
            "rephrase_answer": rephrase_answer,
            "locality_answer": locality_answer,
            "target_hit": target_hit,
            "ground_truth_hit": ground_truth_hit,
            "locality_hit": locality_hit,
        }

        records.append(record)

        # 4. 打印当前样本的完整回答
        print()
        print("-" * 100)
        print(f"Case {idx} / {len(facts)}")
        print(f"Subject: {fact.get('subject', 'N/A')}")
        print()

        print("[Direct Prompt]")
        print(f"Prompt: {fact['prompt']}")
        print(f"Model Answer: {answer}")
        print(f"Target New: {fact['target_new']}")
        print(f"Ground Truth / Old Answer: {fact['ground_truth']}")
        print(f"Target Hit: {target_hit}")
        print(f"Old Knowledge Hit: {ground_truth_hit}")
        print()

        print("[Rephrase Prompt]")
        print(f"Prompt: {fact['rephrase_prompt']}")
        print(f"Model Answer: {rephrase_answer}")
        print(f"Target New: {fact['target_new']}")
        print(f"Rephrase Target Hit: {contains(rephrase_answer, fact['target_new'])}")
        print()

        print("[Locality Prompt]")
        print(f"Prompt: {fact['locality_prompt']}")
        print(f"Model Answer: {locality_answer}")
        print(f"Expected Locality Answer: {fact['locality_ground_truth']}")
        print(f"Locality Hit: {locality_hit}")
        print("-" * 100)

    # 5. 汇总统计指标
    n = len(records)

    metrics = {
        "cases": n,
        "target_hit_rate": round(sum(r["target_hit"] for r in records) / n, 4),
        "old_knowledge_hit_rate": round(sum(r["ground_truth_hit"] for r in records) / n, 4),
        "locality_generation_rate": round(sum(r["locality_hit"] for r in records) / n, 4),
        "rephrase_target_hit_rate": round(
            sum(contains(r["rephrase_answer"], r["target_new"]) for r in records) / n, 4
        ),
    }

    # 6. 统计运行耗时和显存
    stats = runtime(start)

    # 7. 保存完整结果到 JSON 文件
    save_json(
        RESULT_DIR / "baseline_results.json",
        {
            "records": records,
            "metrics": metrics,
            "runtime": stats,
        },
    )

    # 8. 终端打印汇总结果
    print()
    print("=" * 100)
    print("Baseline Summary")
    print("=" * 100)
    print(f"cases: {metrics['cases']}")
    print(f"target answer hit before editing: {metrics['target_hit_rate'] * 100:.1f}%")
    print(f"old answer hit before editing: {metrics['old_knowledge_hit_rate'] * 100:.1f}%")
    print(f"rephrase target hit before editing: {metrics['rephrase_target_hit_rate'] * 100:.1f}%")
    print(f"locality generation sanity check: {metrics['locality_generation_rate'] * 100:.1f}%")
    print(f"elapsed seconds: {stats['elapsed_seconds']}")
    print(f"peak memory MB: {stats['peak_memory_mb']}")
    print("saved: artifacts/results/baseline_results.json")
    print("=" * 100)

    # 9. 写入日志文件
    write_log(
        "baseline",
        [
            "Task 1 Baseline Evaluation (Qwen2.5-0.5B-Instruct)",
            f"cases: {metrics['cases']}",
            f"target answer hit before editing: {metrics['target_hit_rate'] * 100:.1f}%",
            f"old answer hit before editing: {metrics['old_knowledge_hit_rate'] * 100:.1f}%",
            f"rephrase target hit before editing: {metrics['rephrase_target_hit_rate'] * 100:.1f}%",
            f"locality generation sanity check: {metrics['locality_generation_rate'] * 100:.1f}%",
            f"elapsed seconds: {stats['elapsed_seconds']}",
            f"peak memory MB: {stats['peak_memory_mb']}",
            "saved: artifacts/results/baseline_results.json",
        ],
    )


if __name__ == "__main__":
    main()