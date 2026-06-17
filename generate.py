import argparse

import torch

from src.tinygpt import CharTokenizer, GPTConfig, TinyGPT


def load_model(checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    config = GPTConfig(**payload["config"])
    model = TinyGPT(config)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/bgpt.pt")
    parser.add_argument("--prompt", default="Zuno helps")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)
    ids = tokenizer.encode(args.prompt)
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
