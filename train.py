import argparse
import json
from pathlib import Path

import torch

from src.tinygpt import CharTokenizer, GPTConfig, TinyGPT


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/business_seed.txt")
    parser.add_argument("--out", default="checkpoints/bgpt.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size)
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(encoded, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0:
            print(f"step {step:04d} loss {loss.item():.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": config.__dict__,
        "tokenizer": tokenizer.to_dict(),
    }, out_path)
    Path(out_path.with_suffix(".json")).write_text(json.dumps({
        "checkpoint": str(out_path),
        "vocab_size": tokenizer.vocab_size,
        "block_size": args.block_size,
        "steps": args.steps,
    }, indent=2), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
