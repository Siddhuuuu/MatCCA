"""
bpe_data.py — same train/val/batch interface as real_data.py, but using a real trained
BPE tokenizer instead of character-level. This is what makes the data pipeline "real"
in the subword-tokenization sense that actual LLMs use, not just real-text-not-toy-data.
"""

import torch
from tokenizers import Tokenizer


class BPETokenizerWrapper:
    """Thin wrapper so this matches CharTokenizer's interface from real_data.py."""

    def __init__(self, tokenizer_path: str):
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tok.encode(text).ids, dtype=torch.long)

    def decode(self, ids) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self.tok.decode(ids)


def prepare_dataset(corpus_path: str, tokenizer_path: str, val_fraction: float = 0.02):
    """val_fraction default lower than the toy pipeline's 0.1 — with real data at real
    scale, 2% is already plenty of held-out tokens for a stable validation estimate,
    and keeps more data in the training split."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    tok = BPETokenizerWrapper(tokenizer_path)
    data = tok.encode(text)
    n = len(data)
    split_idx = int(n * (1 - val_fraction))
    train_data, val_data = data[:split_idx], data[split_idx:]
    return train_data, val_data, tok


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device="cpu"):
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in starts])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, batch_size, seq_len, device="cpu", n_batches=20, **attn_kwargs):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, seq_len, device)
        _, loss = model(x, targets=y, **attn_kwargs)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "shakespeare.txt"
    tok_path = sys.argv[2] if len(sys.argv) > 2 else "shakespeare_tokenizer.json"

    train_data, val_data, tok = prepare_dataset(corpus, tok_path)
    print(f"vocab_size: {tok.vocab_size}")
    print(f"train: {len(train_data):,} tokens, val: {len(val_data):,} tokens")

    x, y = get_batch(train_data, batch_size=4, seq_len=32)
    print(f"\nsample batch: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}")
    print(f"x[0] decoded: {tok.decode(x[0])!r}")
    print(f"y[0] decoded: {tok.decode(y[0])!r}  <- should be x[0] shifted one token left")
