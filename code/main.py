from __future__ import annotations

import argparse
import glob
import math
import os
import torch

import data
import lm
from torch import optim
from transformer import TransformerLM


def tokenizer_to_state(tokenizer: data.CharTokenizer) -> dict:
    return {
        "vocab": tokenizer.vocab,
        "symbols": tokenizer.symbols,
    }


def tokenizer_from_state(state: dict) -> data.CharTokenizer:
    tokenizer = data.CharTokenizer()
    tokenizer.vocab = list(state["vocab"])
    tokenizer.symbols = list(state.get("symbols", ["<PAD>"]))
    tokenizer.tokens = set(tokenizer.vocab) - set(tokenizer.symbols)
    tokenizer.stoi = {s: i for i, s in enumerate(tokenizer.vocab)}
    return tokenizer


def tokenize_with_existing_tokenizer(tokenizer: data.CharTokenizer, data_path: str) -> list[list[int]]:
    tokenized_data: list[list[int]] = []
    for fname in glob.glob(f"{data_path}/*.txt"):
        with open(fname, encoding="utf-8") as fh:
            text = fh.read()
            tokenized_data.append(tokenizer.tokenize(text))
    return tokenized_data


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    tokenizer: data.CharTokenizer,
    num_batches: int,
    model_config: dict,
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer_state": tokenizer_to_state(tokenizer),
        "num_batches": num_batches,
        "model_config": model_config,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def compute_learning_rate(
    current_batch: int,
    total_batches: int,
    base_lr: float,
    min_lr: float,
    warmup_batches: int,
    schedule: str,
) -> float:
    if schedule == "constant":
        return base_lr

    min_lr = min(min_lr, base_lr)

    if warmup_batches > 0 and current_batch < warmup_batches:
        return base_lr * float(current_batch + 1) / float(warmup_batches)

    decay_steps = max(1, total_batches - warmup_batches)
    decay_position = max(0, current_batch - warmup_batches)
    progress = min(1.0, float(decay_position) / float(decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny transformer language model")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-path", type=str, default="../data/en/")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--embed-size", type=int, default=192)
    parser.add_argument("--mlp-hidden-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-5)
    parser.add_argument("--lr-warmup-batches", type=int, default=2000)
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "warmup-cosine"], default="warmup-cosine")
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--num-batches-to-train", type=int, default=50000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--sample-prefix", type=str, default="Hello")
    parser.add_argument("--sample-len", type=int, default=500)
    parser.add_argument("--use-better-sampling",default=True, action="store_true", help="Use temperature and top-k sampling instead of basic sampling")
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--sample-topk", type=int, default=5)

    parser.add_argument("--checkpoint-dir", type=str, default="../checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--override-lr", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp_hidden_size = args.mlp_hidden_size if args.mlp_hidden_size is not None else args.embed_size * 4

    tokenizer, tokenized_data = data.load_data(args.data_path)

    model_config = {
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "embed_size": args.embed_size,
        "max_context_len": args.seq_len,
        "vocab_size": tokenizer.vocab_size(),
        "mlp_hidden_size": mlp_hidden_size,
        "with_residuals": True,
        "pre_norm": True,
    }

    model: torch.nn.Module = TransformerLM(**model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=[0.9, 0.95])
    start_batch = 0

    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)

        checkpoint_model_config = checkpoint.get("model_config", model_config)
        model = TransformerLM(**checkpoint_model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model_config = checkpoint_model_config

        if "tokenizer_state" in checkpoint:
            tokenizer = tokenizer_from_state(checkpoint["tokenizer_state"])
            tokenized_data = tokenize_with_existing_tokenizer(tokenizer, args.data_path)

        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=[0.9, 0.95])

        if args.resume_training:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_batch = int(checkpoint.get("num_batches", 0))

        print(f"Loaded checkpoint from {args.load_checkpoint}")

    if args.override_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = args.override_lr
        print(f"Overrode optimizer learning rate to {args.override_lr}")

    # Data items are one token longer than model context for next-token labels.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, model_config["max_context_len"] + 1))

    model.train()
    num_batches = start_batch

    if args.override_lr is not None:
        current_lr = args.override_lr
    else:
        current_lr = compute_learning_rate(
            current_batch=num_batches,
            total_batches=args.num_batches_to_train,
            base_lr=args.learning_rate,
            min_lr=args.min_learning_rate,
            warmup_batches=args.lr_warmup_batches,
            schedule=args.lr_schedule,
        )

    for group in optimizer.param_groups:
        group["lr"] = current_lr

    while num_batches < args.num_batches_to_train:
        for batch in data.batch_items(data_iter, args.batch_size):
            if num_batches >= args.num_batches_to_train:
                break

            if args.override_lr is None:
                current_lr = compute_learning_rate(
                    current_batch=num_batches,
                    total_batches=args.num_batches_to_train,
                    base_lr=args.learning_rate,
                    min_lr=args.min_learning_rate,
                    warmup_batches=args.lr_warmup_batches,
                    schedule=args.lr_schedule,
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()

            num_batches += 1

            if args.log_every > 0 and num_batches % args.log_every == 0:
                print(f"Seen {num_batches} batches on {device}. lr={current_lr:.8f}. last loss is: {loss.item()}")

            if args.sample_every > 0 and num_batches % args.sample_every == 0:
                model.eval()
                if args.use_better_sampling:
                    sampled = tokenizer.detokenize(
                        model.better_sample_continuation(
                            tokenizer.tokenize(args.sample_prefix),
                            args.sample_len,
                            temperature=args.sample_temperature,
                            topK=args.sample_topk
                        )
                    )
                else:
                    sampled = tokenizer.detokenize(
                        model.sample_continuation(tokenizer.tokenize(args.sample_prefix), args.sample_len)
                    )
                model.train()
                print(f"Model sample: '''{sampled}'''")
                print("")

            if args.checkpoint_every > 0 and num_batches % args.checkpoint_every == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{num_batches}.pt")
                save_checkpoint(checkpoint_path, model, optimizer, tokenizer, num_batches, model_config)

    final_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_last.pt")
    save_checkpoint(final_checkpoint_path, model, optimizer, tokenizer, num_batches, model_config)


if __name__ == "__main__":
    main()
