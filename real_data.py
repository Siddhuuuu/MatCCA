"""
real_data.py — a real text data pipeline, no toy periodic patterns anywhere.

Character-level tokenizer, deliberately: it needs zero external downloads (unlike a
BPE tokenizer like tiktoken, which needs to fetch vocab files from OpenAI's servers —
that call is blocked in the sandbox this was built in, so it's untestable there, but
also just an unnecessary dependency for a first real-data run). A char-level vocab is
tiny (~65-100 symbols for English text), which keeps the embedding/output layers small
so more of the parameter budget goes toward the attention mechanism differences you're
actually trying to measure, not a giant 50k-word embedding table. This is the same
choice nanoGPT's own from-scratch demo makes, for the same reasons.

Point this at ANY .txt file. Tested in-sandbox against the tiny-Shakespeare corpus
(~1MB, real text, the same one nanoGPT's README demo uses) pulled directly from
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import torch


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Builds its vocabulary directly from the data — no downloaded files, no
    external dependency, fully deterministic given the same text."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, ids) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


def prepare_dataset(path: str, val_fraction: float = 0.1):
    """Load a text file, tokenize it, split into train/val by position (not
    shuffled — keeps validation genuinely held-out, not just held-out sentences
    interleaved with training ones)."""
    text = load_text(path)
    tok = CharTokenizer(text)
    data = tok.encode(text)
    n = len(data)
    split_idx = int(n * (1 - val_fraction))
    train_data, val_data = data[:split_idx], data[split_idx:]
    return train_data, val_data, tok


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device="cpu"):
    """Sample random contiguous windows — standard nanoGPT-style batching for a
    single long token stream. Requires len(data) > seq_len."""
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
    # Self-test against real text — run this file directly to sanity check the
    # pipeline before wiring it into actual training.
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "shakespeare.txt"
    train_data, val_data, tok = prepare_dataset(path)
    print(f"loaded '{path}': {len(train_data) + len(val_data):,} characters total")
    print(f"vocab size: {tok.vocab_size} unique characters")
    print(f"train: {len(train_data):,} chars, val: {len(val_data):,} chars")
    print(f"\nfirst 200 chars decoded back from encoded ids (round-trip check):")
    print(repr(tok.decode(train_data[:200])))
    assert tok.decode(train_data[:200]) == load_text(path)[:200], "round-trip encode/decode mismatch!"
    print("\nround-trip OK.")

    x, y = get_batch(train_data, batch_size=4, seq_len=32)
    print(f"\nsample batch: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}")
    print(f"x[0] decoded: {repr(tok.decode(x[0]))}")
    print(f"y[0] decoded: {repr(tok.decode(y[0]))}  <- should be x[0] shifted left by one char")
