"""
download_tinystories.py — pulls the TinyStories dataset and saves it as plain .txt
files for train_tokenizer.py / bpe_data.py to consume.

HONESTY NOTE: this specific script could NOT be tested in the sandbox this was built
in — it has no route to huggingface.co. It follows the standard, well-documented
`datasets` library pattern exactly, but you are the first one actually running it.
If something breaks, the error message will tell us exactly what to fix — paste it
back the same way as always.

TinyStories (Eldan & Li, 2023) was purpose-built for training tiny language models —
short, simple stories with coherent structure, specifically so small models can learn
to produce fluent text. That's exactly the right fit for Kaggle-scale compute, and a
better choice than a C4 subset (which is much messier/more diverse, needing far more
scale to learn well).

Run this ONCE on Kaggle (needs internet enabled in notebook settings), then everything
downstream (tokenizer training, model training) uses the local .txt files it produces.
"""

import argparse
from datasets import load_dataset


def download_tinystories(out_train="tinystories_train.txt", out_val="tinystories_val.txt",
                          max_train_examples=200_000, max_val_examples=2_000):
    print("Downloading TinyStories from Hugging Face (roneneldan/TinyStories)...")
    print("This needs internet enabled in your Kaggle notebook's settings panel.")

    ds = load_dataset("roneneldan/TinyStories")

    print(f"Writing up to {max_train_examples:,} training stories to '{out_train}'...")
    with open(out_train, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds["train"]):
            if i >= max_train_examples:
                break
            f.write(example["text"].strip() + "\n\n")

    print(f"Writing up to {max_val_examples:,} validation stories to '{out_val}'...")
    with open(out_val, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds["validation"]):
            if i >= max_val_examples:
                break
            f.write(example["text"].strip() + "\n\n")

    import os
    train_size = os.path.getsize(out_train) / 1e6
    val_size = os.path.getsize(out_val) / 1e6
    print(f"\nDone. '{out_train}': {train_size:.1f} MB, '{out_val}': {val_size:.1f} MB")
    print("Next: python train_tokenizer.py tinystories_train.txt tokenizer.json --vocab_size 8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_examples", type=int, default=200_000)
    parser.add_argument("--max_val_examples", type=int, default=2_000)
    args = parser.parse_args()
    download_tinystories(max_train_examples=args.max_train_examples, max_val_examples=args.max_val_examples)
