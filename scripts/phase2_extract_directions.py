#!/usr/bin/env python3
"""Extract and evaluate contrastive activation directions for Phase 2.

The script is independent of the representation-engineering checkout. It reads
the project's religious/neutral and secular-moral/neutral JSONL datasets,
formats prompts with the selected tokenizer's chat template, caches final-token
activations for each transformer layer, computes mean paired-difference
directions, and evaluates one binary logistic-regression probe per layer.

Optional refusal data can be supplied as JSONL. Each row must either contain
``harmful_question`` and ``harmless_question`` fields (with an optional
``split``), or ``text``, ``label`` (``harmful``/``harmless`` or 1/0), and an
optional ``split`` field. Include both ``train`` and ``validation`` rows; a
missing split defaults to ``train``.

Example:
    python scripts/phase2_extract_directions.py \
      --model google/gemma-2-9b-it \
      --refusal-data data/refusal_prompts.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("phase2")


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def split_name(row: dict[str, Any]) -> str:
    split = str(row.get("split", "train")).lower()
    if split in {"val", "valid", "validation", "dev"}:
        return "validation"
    if split == "train":
        return "train"
    raise ValueError(f"Unsupported split {split!r}; use train or validation")


def make_pair_examples(
    rows: list[dict[str, Any]], positive_key: str, negative_key: str
) -> tuple[list[str], list[str], list[str]]:
    """Return flattened texts, per-text labels, and per-pair splits."""
    texts: list[str] = []
    labels: list[str] = []
    pair_splits: list[str] = []
    for i, row in enumerate(rows):
        positive, negative = row.get(positive_key), row.get(negative_key)
        if not isinstance(positive, str) or not isinstance(negative, str):
            raise ValueError(
                f"Row {i + 1}: expected string fields {positive_key!r} and {negative_key!r}"
            )
        texts.extend((positive, negative))
        labels.extend(("positive", "negative"))
        pair_splits.append(split_name(row))
    return texts, labels, pair_splits


def make_refusal_examples(rows: list[dict[str, Any]]) -> tuple[list[str], list[int], list[str]]:
    texts: list[str] = []
    labels: list[int] = []
    splits: list[str] = []
    for i, row in enumerate(rows):
        split = split_name(row)
        harmful = row.get("harmful_question", row.get("harmful"))
        harmless = row.get("harmless_question", row.get("harmless"))
        if isinstance(harmful, str) and isinstance(harmless, str):
            texts.extend((harmful, harmless))
            labels.extend((1, 0))
            splits.extend((split, split))
            continue

        text = row.get("text")
        label = row.get("label")
        if not isinstance(text, str):
            raise ValueError(
                f"Refusal row {i + 1}: expected paired harmful/harmless fields or a text field"
            )
        if isinstance(label, str):
            normalized = label.strip().lower()
            if normalized in {"harmful", "harm", "unsafe", "1"}:
                label = 1
            elif normalized in {"harmless", "safe", "benign", "0"}:
                label = 0
        if label not in (0, 1):
            raise ValueError(f"Refusal row {i + 1}: label must identify harmful=1 or harmless=0")
        texts.append(text)
        labels.append(int(label))
        splits.append(split)

    if len(set(labels)) != 2:
        raise ValueError("Refusal data must contain both harmful and harmless examples")
    return texts, labels, splits


def format_prompt(tokenizer, text: str, use_chat_template: bool, generation_prompt: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=generation_prompt,
        )
    return text


def fingerprint(model_name: str, prompts: list[str], max_length: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(str(max_length).encode("ascii"))
    for prompt in prompts:
        hasher.update(b"\0")
        hasher.update(prompt.encode("utf-8"))
    return hasher.hexdigest()


def save_activation_cache(cache_base: Path, cache_key: str, activations: dict[int, np.ndarray]) -> None:
    cache_base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_base.with_suffix(".npz"), **{
        f"layer_{layer}": value for layer, value in activations.items()
    })
    cache_base.with_suffix(".json").write_text(
        json.dumps({"fingerprint": cache_key, "layers": sorted(activations)}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_activation_cache(cache_base: Path, cache_key: str) -> Optional[dict[int, np.ndarray]]:
    metadata_path, archive_path = cache_base.with_suffix(".json"), cache_base.with_suffix(".npz")
    if not metadata_path.exists() or not archive_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != cache_key:
            return None
        with np.load(archive_path) as archive:
            return {
                int(key.removeprefix("layer_")): archive[key].copy()
                for key in archive.files
            }
    except (OSError, ValueError, json.JSONDecodeError):
        LOG.warning("Ignoring unreadable activation cache %s", archive_path)
        return None


def collect_activations(
    model,
    tokenizer,
    texts: list[str],
    model_name: str,
    cache_base: Path,
    batch_size: int,
    max_length: int,
) -> dict[int, np.ndarray]:
    key = fingerprint(model_name, texts, max_length)
    cached = load_activation_cache(cache_base, key)
    if cached is not None:
        LOG.info("Using cached activations: %s", cache_base.with_suffix(".npz"))
        return cached

    embeddings = model.get_input_embeddings().weight
    input_device = embeddings.device
    if input_device.type == "meta":
        input_device = model.device
    chunks: dict[int, list[np.ndarray]] = {}
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = {key: value.to(input_device) for key, value in tokens.items()}
        positions = torch.arange(tokens["attention_mask"].shape[1], device=input_device)
        last_positions = (
            tokens["attention_mask"].long() * positions.unsqueeze(0)
        ).max(dim=1).values

        with torch.inference_mode():
            output = model(**tokens, output_hidden_states=True, use_cache=False)
        if output.hidden_states is None:
            raise RuntimeError("Model did not return hidden states despite output_hidden_states=True")
        for layer in range(1, len(output.hidden_states)):
            states = output.hidden_states[layer]
            rows = torch.arange(states.shape[0], device=states.device)
            final_states = states[rows, last_positions.to(states.device)].float().cpu().numpy()
            chunks.setdefault(layer, []).append(final_states)
        del output, tokens

    activations = {layer: np.concatenate(parts, axis=0) for layer, parts in chunks.items()}
    save_activation_cache(cache_base, key, activations)
    LOG.info("Saved activation cache: %s", cache_base.with_suffix(".npz"))
    return activations


def paired_direction(
    activations: dict[int, np.ndarray], pair_splits: list[str]
) -> dict[int, np.ndarray]:
    train_indices = np.flatnonzero(np.asarray(pair_splits) == "train")
    if not len(train_indices):
        raise ValueError("No training pairs found")
    vectors = {}
    for layer, values in activations.items():
        paired = values.reshape(-1, 2, values.shape[-1])
        vectors[layer] = (paired[train_indices, 0] - paired[train_indices, 1]).mean(axis=0)
    return vectors


def probe_metrics(
    activations: dict[int, np.ndarray], labels: list[int], splits: list[str]
) -> dict[int, dict[str, float]]:
    y = np.asarray(labels, dtype=np.int64)
    split_array = np.asarray(splits)
    train_mask, val_mask = split_array == "train", split_array == "validation"
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Each contrast needs examples in both train and validation splits")
    if len(set(y[train_mask])) != 2 or len(set(y[val_mask])) != 2:
        raise ValueError("Both classes must appear in train and validation splits")

    metrics = {}
    for layer, values in activations.items():
        probe = LogisticRegression(solver="liblinear", max_iter=2000, random_state=0)
        probe.fit(values[train_mask], y[train_mask])
        prediction = probe.predict(values[val_mask])
        metrics[layer] = {
            "accuracy": float(accuracy_score(y[val_mask], prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y[val_mask], prediction)),
        }
    return metrics


def pair_labels_and_splits(pair_splits: list[str], positive_label: int = 1) -> tuple[list[int], list[str]]:
    labels, splits = [], []
    for split in pair_splits:
        labels.extend((positive_label, 1 - positive_label))
        splits.extend((split, split))
    return labels, splits


def select_layer(metrics: dict[int, dict[str, float]]) -> int:
    return max(metrics, key=lambda layer: (metrics[layer]["balanced_accuracy"], metrics[layer]["accuracy"]))


def cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0 or not math.isfinite(denominator):
        return None
    return float(np.dot(a, b) / denominator)


def save_results(
    out_dir: Path,
    directions: dict[str, dict[int, np.ndarray]],
    metrics: dict[str, dict[int, dict[str, float]]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{name}_layer_{layer}": vector.astype(np.float32)
        for name, by_layer in directions.items()
        for layer, vector in by_layer.items()
    }
    np.savez_compressed(out_dir / "directions.npz", **arrays)

    summary: dict[str, Any] = {"datasets": {}}
    for name, layer_metrics in metrics.items():
        best = select_layer(layer_metrics)
        summary["datasets"][name] = {
            "best_validation_layer": best,
            "best_validation_balanced_accuracy": layer_metrics[best]["balanced_accuracy"],
            "best_validation_accuracy": layer_metrics[best]["accuracy"],
            "per_layer_probe_metrics": {str(k): v for k, v in layer_metrics.items()},
        }
    summary["directions"] = {
        name: {str(layer): {"l2_norm": float(np.linalg.norm(vec))} for layer, vec in by_layer.items()}
        for name, by_layer in directions.items()
    }
    summary["cosine_similarities"] = {}
    shared_names = list(directions)
    if len(shared_names) > 1:
        shared_layers = sorted(set.intersection(*(set(directions[name]) for name in shared_names)))
        for i, first in enumerate(shared_names):
            for second in shared_names[i + 1 :]:
                summary["cosine_similarities"][f"{first}_vs_{second}"] = {
                    str(layer): cosine(directions[first][layer], directions[second][layer])
                    for layer in shared_layers
                }
    (out_dir / "phase2_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    csv_path = out_dir / "probe_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as stream:
        stream.write("dataset,layer,validation_accuracy,validation_balanced_accuracy\n")
        for name, layer_metrics in metrics.items():
            for layer, values in sorted(layer_metrics.items()):
                stream.write(f"{name},{layer},{values['accuracy']:.6f},{values['balanced_accuracy']:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint path")
    parser.add_argument(
        "--religious-data", type=Path,
        default=ROOT / "data/religious_neutral_question_pairs.jsonl",
    )
    parser.add_argument(
        "--moral-data", type=Path,
        default=ROOT / "data/secular_moral_neutral_question_pairs.jsonl",
    )
    parser.add_argument("--refusal-data", type=Path, help="Optional harmful/harmless prompt JSONL")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/phase2")
    parser.add_argument("--cache-dir", type=Path, help="Defaults to <out-dir>/activation_cache")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-chat-template", action="store_true", help="Tokenize dataset text as-is")
    parser.add_argument(
        "--no-generation-prompt", action="store_true",
        help="Do not append the model's assistant-generation prefix to chat prompts",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    if not args.religious_data.exists():
        raise FileNotFoundError(args.religious_data)
    if not args.moral_data.exists():
        raise FileNotFoundError(args.moral_data)
    cache_dir = args.cache_dir or args.out_dir / "activation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    LOG.info("Loading model and tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad token or an EOS token for batching")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    use_chat = not args.no_chat_template and bool(getattr(tokenizer, "chat_template", None))
    if not use_chat:
        LOG.warning("No chat template will be applied to prompts")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device_map
    )
    model.eval()
    cache_model_id = (
        f"{args.model}|dtype={args.dtype}|revision={getattr(model.config, '_commit_hash', None)}"
    )

    directions: dict[str, dict[int, np.ndarray]] = {}
    metrics: dict[str, dict[int, dict[str, float]]] = {}
    datasets = [
        ("religiosity", args.religious_data, "religious_question", "neutral_question"),
        ("moral_tone", args.moral_data, "moral_question", "neutral_question"),
    ]

    for name, path, positive_key, negative_key in datasets:
        rows = jsonl_rows(path)
        texts, _, pair_splits = make_pair_examples(rows, positive_key, negative_key)
        prompts = [
            format_prompt(tokenizer, text, use_chat, not args.no_generation_prompt)
            for text in texts
        ]
        activations = collect_activations(
            model, tokenizer, prompts, cache_model_id, cache_dir / name,
            args.batch_size, args.max_length,
        )
        directions[name] = paired_direction(activations, pair_splits)
        probe_labels, probe_splits = pair_labels_and_splits(pair_splits)
        metrics[name] = probe_metrics(activations, probe_labels, probe_splits)
        best = select_layer(metrics[name])
        LOG.info(
            "%s: best validation layer %s (balanced accuracy %.3f)",
            name, best, metrics[name][best]["balanced_accuracy"],
        )

    if args.refusal_data:
        rows = jsonl_rows(args.refusal_data)
        texts, labels, splits = make_refusal_examples(rows)
        prompts = [
            format_prompt(tokenizer, text, use_chat, not args.no_generation_prompt)
            for text in texts
        ]
        activations = collect_activations(
            model, tokenizer, prompts, cache_model_id, cache_dir / "refusal",
            args.batch_size, args.max_length,
        )
        split_array = np.asarray(splits)
        train_mask = split_array == "train"
        if len(set(np.asarray(labels)[train_mask])) != 2:
            raise ValueError("Refusal training split must contain harmful and harmless examples")
        labels_array = np.asarray(labels)
        directions["refusal"] = {
            layer: values[train_mask & (labels_array == 1)].mean(axis=0)
            - values[train_mask & (labels_array == 0)].mean(axis=0)
            for layer, values in activations.items()
        }
        metrics["refusal"] = probe_metrics(activations, labels, splits)
        best = select_layer(metrics["refusal"])
        LOG.info(
            "refusal: best validation layer %s (balanced accuracy %.3f)",
            best, metrics["refusal"][best]["balanced_accuracy"],
        )
    else:
        LOG.warning("No --refusal-data supplied; refusal direction and its cosine comparisons are skipped")

    save_results(args.out_dir, directions, metrics)
    LOG.info("Wrote directions, probe metrics, and summary to %s", args.out_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.error("%s", exc)
        raise
