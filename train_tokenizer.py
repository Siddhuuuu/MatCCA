"""
train_tokenizer.py — trains a real BPE (subword) tokenizer from scratch on your corpus.

Deliberately a SMALL vocab (default 8000), not GPT-2's 50257. Reason: with weight-tying,
the embedding table alone would be vocab_size x dim parameters — at 50257 x 512 that's
~26M params, which would swamp the entire rest of the model and dilute the actual
attention-mechanism comparison you care about (MHA vs GQA vs CCA vs MatCCA). An 8000-vocab
BPE tokenizer keeps real subword structure (unlike char-level) while keeping the embedding
table a sane fraction of total params. This is a standard tradeoff in small-scale LM
research, not a shortcut.

Usage: python train_tokenizer.py corpus.txt tokenizer.json --vocab_size 8000
"""

import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def train_bpe_tokenizer(corpus_path: str, save_path: str, vocab_size: int = 8000):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        show_progress=True,
    )
    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_path")
    parser.add_argument("save_path")
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()

    print(f"Training BPE tokenizer, vocab_size={args.vocab_size}, on '{args.corpus_path}'...")
    tok = train_bpe_tokenizer(args.corpus_path, args.save_path, args.vocab_size)
    print(f"Saved to '{args.save_path}'. Actual vocab size: {tok.get_vocab_size()}")

    with open(args.corpus_path, "r", encoding="utf-8") as f:
        sample = f.read()[:300]
    encoded = tok.encode(sample)
    print(f"\nRound-trip sanity check on first 300 chars:")
    print(f"  original : {sample[:100]!r}...")
    print(f"  token IDs: {encoded.ids[:20]}... ({len(encoded.ids)} tokens for {len(sample)} chars, "
          f"compression ~{len(sample) / max(len(encoded.ids), 1):.2f}x chars/token)")
    decoded = tok.decode(encoded.ids)
    print(f"  decoded  : {decoded[:100]!r}...")
    assert decoded.strip() == sample.strip() or decoded == sample, "round-trip mismatch (check whitespace handling)"
    print("  round-trip OK.")
