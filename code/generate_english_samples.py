from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch

from main import tokenizer_from_state
from transformer import TransformerLM


SHAKESPEARE_PROMPTS = [
    "KING:",
    "QUEEN:",
    "LORD:",
    "SIR:",
    "HAMLET:",
    "ROMEO:",
    "JULIET:",
    "MACBETH:",
    
    "KING:\nWhat news from the battle?",
    "QUEEN:\nMy lord, I fear the worst, for",
    "HAMLET:\nTo be or not to be, I",
    "ROMEO:\nO gentle Juliet, if thou",
    "JULIET:\nMy only love sprung from",
    
    "SOLDIER:\nThe enemy approaches and",
    "MESSENGER:\nMy lord, I bring grave tidings",
    "SERVANT:\nThe guests await your word",
    
    "KING:\nWhy dost thou betray me thus?",
    "QUEEN:\nSpeak plainly, for I cannot",
    "DUKE:\nThis matter grows too dangerous",
    
    "FIRST LORD:\nHe hath conspired against us",
    "SECOND LORD:\nI like not this silence",
    
    "PRINCE:\nIf fate be kind, then let",
    "LADY:\nOut, damned spot! Out, I say",
    
    "CAPTAIN:\nThe battle is lost unless",
    "KNIGHT:\nI swear upon my honor",
    
    "FOOL:\nWhy, this is a merry jest, yet",
    
    "KING:\nCall forth the council, for",
    "QUEEN:\nThe night is dark and full of",
    
    "ROMEO:\nWith love's light wings did I",
    "JULIET:\nParting is such sweet sorrow that",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 50 English samples from the best available checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="english-sample.txt")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--sample-len", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_best_checkpoint(checkpoint: str | None, checkpoint_dir: str) -> Path:
    if checkpoint is not None:
        selected = Path(checkpoint)
        if not selected.exists():
            raise FileNotFoundError(f"Checkpoint not found: {selected}")
        return selected

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    last_ckpt = ckpt_dir / "checkpoint_last.pt"
    if last_ckpt.exists():
        return last_ckpt

    step_checkpoints = []
    pattern = re.compile(r"checkpoint_step_(\d+)\.pt$")
    for path in ckpt_dir.glob("checkpoint_step_*.pt"):
        match = pattern.search(path.name)
        if match:
            step_checkpoints.append((int(match.group(1)), path))

    if not step_checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir}. Expected checkpoint_last.pt or checkpoint_step_*.pt"
        )

    step_checkpoints.sort(key=lambda x: x[0])
    return step_checkpoints[-1][1]


def safe_prompt(prompt: str, vocab_lookup: dict[str, int], fallback_token: str) -> str:
    filtered = "".join(ch for ch in prompt if ch in vocab_lookup)
    if filtered:
        return filtered
    return fallback_token


def pick_fallback_token(vocab: list[str], symbols: list[str]) -> str:
    symbol_set = set(symbols)
    for token in vocab:
        if token not in symbol_set and len(token) == 1:
            return token
    for token in vocab:
        if token not in symbol_set:
            return token
    raise ValueError("Tokenizer vocabulary has no non-symbol tokens.")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    checkpoint_path = find_best_checkpoint(args.checkpoint, args.checkpoint_dir)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    tokenizer = tokenizer_from_state(checkpoint["tokenizer_state"])
    model = TransformerLM(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fallback_token = pick_fallback_token(tokenizer.vocab, tokenizer.symbols)
    prompts = SHAKESPEARE_PROMPTS[:]

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    samples: list[str] = []
    for i in range(args.num_samples):
        prompt = prompts[i % len(prompts)]
        prompt_for_model = safe_prompt(prompt, tokenizer.stoi, fallback_token)
        prefix_tokens = tokenizer.tokenize(prompt_for_model)

        generated_tokens = model.better_sample_continuation(
            prefix_tokens,
            max_tokens_to_generate=args.sample_len,
            temperature=args.temperature,
            topK=args.topk,
        )
        generated_text = tokenizer.detokenize(generated_tokens)

        sample_block = (
            f"Sample {i + 1}\n"
            f"Prompt: {prompt}\n"
            f"Generated: {generated_text}"
        )
        samples.append(sample_block)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(samples) + "\n", encoding="utf-8")

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Wrote {len(samples)} samples to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
