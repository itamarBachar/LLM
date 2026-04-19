from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import data
from transformer import TransformerLM


def tokenizer_from_state(state: dict) -> data.CharTokenizer:
    tokenizer = data.CharTokenizer()
    tokenizer.vocab = list(state["vocab"])
    tokenizer.symbols = list(state.get("symbols", ["<PAD>"]))
    tokenizer.tokens = set(tokenizer.vocab) - set(tokenizer.symbols)
    tokenizer.stoi = {s: i for i, s in enumerate(tokenizer.vocab)}
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run samples through a trained LM, save attention matrices, and produce visualizations/statistics."
    )
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/checkpoint_last.pt")
    parser.add_argument("--output-dir", type=str, default="../attention_analysis")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto if omitted")

    parser.add_argument("--sample", action="append", default=[], help="A text sample. Can be repeated.")
    parser.add_argument("--samples-file", type=str, default=None, help="Path to UTF-8 text file with one sample per line.")
    parser.add_argument("--data-path", type=str, default=None, help="Corpus directory (*.txt) used for random sample extraction.")
    parser.add_argument("--random-samples", type=int, default=0, help="Number of random corpus slices to draw.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-sample-len", type=int, default=None, help="Clip sample length to this many tokens.")
    parser.add_argument("--max-layer-head-heatmaps", type=int, default=2, help="Heads per layer to render as detailed heatmaps.")
    parser.add_argument(
        "--max-samples-with-heatmaps",
        type=int,
        default=5,
        help="Render detailed per-sample heatmaps only for the first N samples.",
    )
    parser.add_argument("--aggregate-len", type=int, default=64, help="Prefix length for aggregate position heatmaps.")
    return parser.parse_args()


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_samples_from_file(path: str) -> list[str]:
    samples: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.rstrip("\n")
            if text.strip():
                samples.append(text)
    return samples


def read_random_corpus_samples(data_path: str, count: int, max_len: int, rng: random.Random) -> list[str]:
    file_paths = sorted(glob.glob(f"{data_path}/*.txt"))
    corpora: list[str] = []
    for fpath in file_paths:
        with open(fpath, "r", encoding="utf-8") as fh:
            text = fh.read()
            if text.strip():
                corpora.append(text)

    if not corpora or count <= 0:
        return []

    samples: list[str] = []
    for _ in range(count):
        text = rng.choice(corpora)
        if len(text) <= max_len:
            samples.append(text)
            continue
        start = rng.randint(0, len(text) - max_len)
        samples.append(text[start : start + max_len])
    return samples


def safe_tokenize(tokenizer: data.CharTokenizer, text: str) -> list[int] | None:
    try:
        return tokenizer.tokenize(text)
    except KeyError as exc:
        missing = str(exc)
        print(f"Skipping sample due to unseen token {missing}: {text[:80]!r}")
        return None


def decode_tokens(tokenizer: data.CharTokenizer, token_ids: Iterable[int]) -> str:
    return tokenizer.detokenize(list(token_ids), keep_symbols=True)


def plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    seq_len = len(labels)
    if seq_len <= 40:
        ticks = np.arange(seq_len)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, fontsize=7, rotation=90)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_distance_profile(profile: np.ndarray, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(profile))
    ax.plot(xs, profile, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Distance d = query_pos - key_pos")
    ax.set_ylabel("Average attention mass")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    rng = random.Random(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint["model_config"]

    tokenizer_state = checkpoint.get("tokenizer_state")
    if tokenizer_state is None:
        raise ValueError("Checkpoint has no tokenizer_state. Re-train or save checkpoint with tokenizer state.")
    tokenizer = tokenizer_from_state(tokenizer_state)

    model = TransformerLM(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_context_len = int(model_config["max_context_len"])
    if args.max_sample_len is None:
        max_sample_len = max_context_len
    else:
        max_sample_len = min(args.max_sample_len, max_context_len)

    samples: list[str] = list(args.sample)
    if args.samples_file:
        samples.extend(read_samples_from_file(args.samples_file))
    if args.data_path and args.random_samples > 0:
        samples.extend(read_random_corpus_samples(args.data_path, args.random_samples, max_sample_len, rng))

    if not samples:
        raise ValueError("No input samples were provided. Use --sample, --samples-file, or --data-path with --random-samples.")

    output_root = args.output_dir
    raw_dir = os.path.join(output_root, "raw")
    heatmap_dir = os.path.join(output_root, "heatmaps")
    aggregate_dir = os.path.join(output_root, "aggregate")
    ensure_dir(output_root)
    ensure_dir(raw_dir)
    ensure_dir(heatmap_dir)
    ensure_dir(aggregate_dir)

    distance_sum = None
    distance_count = None

    # Average attention on aligned positions for the first aggregate_len tokens.
    aggregate_len = min(args.aggregate_len, max_sample_len)
    pos_sum = None
    pos_count = 0

    token_pair_sum: dict[tuple[int, int, str, str], float] = defaultdict(float)

    sample_index = 0
    summary_rows: list[dict[str, object]] = []

    for raw_text in samples:
        token_ids = safe_tokenize(tokenizer, raw_text)
        if token_ids is None:
            continue
        if not token_ids:
            continue

        token_ids = token_ids[:max_sample_len]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, attn_by_layer = model(input_ids, return_attn=True)

        # Stack into [L, H, N, N] for this single sample.
        attn = torch.stack(attn_by_layer, dim=0).squeeze(1).detach().cpu()
        logits = logits.detach().cpu()

        seq_tokens = token_ids
        seq_symbols = [tokenizer.vocab[t] for t in seq_tokens]
        predicted_next_ids = logits[0].argmax(dim=-1).tolist()
        predicted_next_text = decode_tokens(tokenizer, predicted_next_ids)

        sample_payload = {
            "text": raw_text,
            "token_ids": seq_tokens,
            "symbols": seq_symbols,
            "predicted_next_token_ids": predicted_next_ids,
            "predicted_next_text": predicted_next_text,
            "attention": attn.to(torch.float32),
        }

        sample_name = f"sample_{sample_index:03d}"
        torch.save(sample_payload, os.path.join(raw_dir, f"{sample_name}.pt"))

        n_layers, n_heads, seq_len, _ = attn.shape

        # Per-sample heatmaps: layer-average over heads + a few specific heads.
        if sample_index < args.max_samples_with_heatmaps:
            for layer in range(n_layers):
                mean_heads = attn[layer].mean(dim=0).numpy()
                plot_heatmap(
                    mean_heads,
                    seq_symbols,
                    title=f"{sample_name} layer={layer} mean(heads)",
                    out_path=os.path.join(heatmap_dir, f"{sample_name}_layer{layer:02d}_mean_heads.png"),
                )
                for head in range(min(n_heads, args.max_layer_head_heatmaps)):
                    head_matrix = attn[layer, head].numpy()
                    plot_heatmap(
                        head_matrix,
                        seq_symbols,
                        title=f"{sample_name} layer={layer} head={head}",
                        out_path=os.path.join(heatmap_dir, f"{sample_name}_layer{layer:02d}_head{head:02d}.png"),
                    )

        # Initialize aggregate tensors lazily once dimensions are known.
        if distance_sum is None:
            distance_sum = torch.zeros((n_layers, n_heads, max_context_len), dtype=torch.float64)
            distance_count = torch.zeros((n_layers, n_heads, max_context_len), dtype=torch.float64)

        # Aggregate by relative distance d = q-k, only for causal positions k <= q.
        q_idx, k_idx = torch.tril_indices(seq_len, seq_len, offset=0)
        d_idx = (q_idx - k_idx).to(torch.int64)
        lower_vals = attn[:, :, q_idx, k_idx].to(torch.float64)
        ones = torch.ones_like(d_idx, dtype=torch.float64)
        for layer in range(n_layers):
            for head in range(n_heads):
                weights = lower_vals[layer, head]
                distance_sum[layer, head, :seq_len] += torch.bincount(d_idx, weights=weights, minlength=seq_len)
                distance_count[layer, head, :seq_len] += torch.bincount(d_idx, weights=ones, minlength=seq_len)

        # Aggregate aligned positions for prefix window.
        if seq_len >= aggregate_len and aggregate_len > 0:
            prefix = attn[:, :, :aggregate_len, :aggregate_len]
            if pos_sum is None:
                pos_sum = torch.zeros_like(prefix, dtype=torch.float64)
            pos_sum += prefix.to(torch.float64)
            pos_count += 1

        # Aggregate token-pair attention behavior.
        query_symbols = [seq_symbols[int(i)] for i in q_idx.tolist()]
        key_symbols = [seq_symbols[int(i)] for i in k_idx.tolist()]
        for layer in range(n_layers):
            for head in range(n_heads):
                weights = lower_vals[layer, head].tolist()
                for idx, w in enumerate(weights):
                    token_pair_sum[(layer, head, query_symbols[idx], key_symbols[idx])] += float(w)

        self_focus = float(attn[:, :, torch.arange(seq_len), torch.arange(seq_len)].mean().item())
        prev_focus = float(attn[:, :, torch.arange(1, seq_len), torch.arange(0, seq_len - 1)].mean().item()) if seq_len > 1 else float("nan")
        first_token_focus = float(attn[:, :, :, 0].mean().item())

        summary_rows.append(
            {
                "sample": sample_name,
                "length": seq_len,
                "self_focus_mean": self_focus,
                "prev_token_focus_mean": prev_focus,
                "first_token_focus_mean": first_token_focus,
                "text_preview": raw_text[:80],
            }
        )

        sample_index += 1
        print(f"Processed {sample_name} (len={seq_len})")

    if sample_index == 0:
        raise ValueError("No valid samples were processed after tokenization checks.")

    with open(os.path.join(output_root, "sample_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample",
                "length",
                "self_focus_mean",
                "prev_token_focus_mean",
                "first_token_focus_mean",
                "text_preview",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    avg_distance = distance_sum / torch.clamp(distance_count, min=1.0)
    n_layers, n_heads, _ = avg_distance.shape

    distance_json: dict[str, list[float]] = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            key = f"layer{layer}_head{head}"
            profile = avg_distance[layer, head].numpy()
            if np.all(distance_count[layer, head].numpy() == 0):
                continue
            distance_json[key] = profile.tolist()
            plot_distance_profile(
                profile,
                title=f"Distance profile layer={layer} head={head}",
                out_path=os.path.join(aggregate_dir, f"distance_profile_layer{layer:02d}_head{head:02d}.png"),
            )

    with open(os.path.join(aggregate_dir, "distance_profiles.json"), "w", encoding="utf-8") as fh:
        json.dump(distance_json, fh)

    if pos_count > 0 and pos_sum is not None:
        pos_avg = pos_sum / float(pos_count)
        for layer in range(n_layers):
            mean_heads = pos_avg[layer].mean(dim=0).numpy()
            labels = [str(i) for i in range(aggregate_len)]
            plot_heatmap(
                mean_heads,
                labels,
                title=f"Aggregate aligned positions layer={layer} mean(heads), n={pos_count}",
                out_path=os.path.join(aggregate_dir, f"aligned_position_layer{layer:02d}_mean_heads.png"),
            )

    # Top token-pair patterns per layer/head.
    with open(os.path.join(aggregate_dir, "top_token_pairs.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["layer", "head", "query_symbol", "key_symbol", "attention_mass"])

        by_head: dict[tuple[int, int], list[tuple[str, str, float]]] = defaultdict(list)
        for (layer, head, q_sym, k_sym), mass in token_pair_sum.items():
            by_head[(layer, head)].append((q_sym, k_sym, mass))

        for (layer, head), rows in sorted(by_head.items()):
            rows.sort(key=lambda item: item[2], reverse=True)
            for q_sym, k_sym, mass in rows[:30]:
                writer.writerow([layer, head, q_sym, k_sym, mass])

    print("Done.")
    print(f"Saved raw matrices to: {raw_dir}")
    print(f"Saved per-sample heatmaps to: {heatmap_dir}")
    print(f"Saved aggregate analyses to: {aggregate_dir}")


if __name__ == "__main__":
    main()
