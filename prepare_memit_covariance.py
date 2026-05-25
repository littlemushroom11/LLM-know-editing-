import argparse
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.real_easyedit import MODEL_PATH, setup_paths, write_log


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare local Wikitext cache and MEMIT covariance stats before editing.")
    parser.add_argument("--mom2-samples", type=int, default=100000)
    parser.add_argument("--layers", default="4,5,6,7")
    parser.add_argument("--dataset-dir", default="data/local_datasets/wikitext-103-raw-v1")
    parser.add_argument("--download-only", action="store_true", help="Only download/save Wikitext; do not compute covariance.")
    return parser.parse_args()


def prepare_dataset(dataset_dir: Path):
    dataset_dir = dataset_dir.resolve()
    os.environ["EASYEDIT_LOCAL_WIKITEXT"] = str(dataset_dir)
    if dataset_dir.exists():
        ds = load_from_disk(str(dataset_dir))
        print(f"Local Wikitext already exists: {dataset_dir}")
        print(ds)
        return dataset_dir

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Wikitext-103 raw dataset...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    print("Saving Wikitext to local disk...")
    ds.save_to_disk(str(dataset_dir))
    print(f"Saved local Wikitext: {dataset_dir}")
    return dataset_dir


def main():
    args = parse_args()
    start = time.perf_counter()
    dataset_dir = prepare_dataset(Path(args.dataset_dir))

    if args.download_only:
        write_log(
            "prepare_memit_covariance",
            [
                "Prepare MEMIT covariance",
                f"local dataset: {dataset_dir}",
                "download_only: true",
                f"elapsed seconds: {round(time.perf_counter() - start, 3)}",
            ],
        )
        return

    setup_paths()
    from easyeditor import MEMITHyperParams
    from easyeditor.models.memit.memit_main import get_cov

    hparams = MEMITHyperParams.from_hparams("hparams_memit_qwen2_0_5b.yaml")
    hparams.layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    hparams.mom2_n_samples = args.mom2_samples

    print("Loading model for covariance preparation...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    ).to(f"cuda:{hparams.device}").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for layer in hparams.layers:
        layer_name = hparams.rewrite_module_tmp.format(layer)
        print(f"Preparing covariance for layer {layer}: {layer_name}")
        cov = get_cov(
            model,
            tok,
            layer_name,
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            force_recompute=False,
            hparams=hparams,
        )
        print(f"Covariance ready for layer {layer}: shape={tuple(cov.shape)}")

    elapsed = round(time.perf_counter() - start, 3)
    peak = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    write_log(
        "prepare_memit_covariance",
        [
            "Prepare MEMIT covariance",
            f"local dataset: {dataset_dir}",
            f"layers: {hparams.layers}",
            f"mom2_samples: {hparams.mom2_n_samples}",
            f"elapsed seconds: {elapsed}",
            f"peak memory MB: {peak}",
            "status: success",
        ],
    )


if __name__ == "__main__":
    main()
