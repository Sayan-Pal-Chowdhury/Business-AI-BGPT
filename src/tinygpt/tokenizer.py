class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)

    def to_dict(self):
        return {"stoi": self.stoi}

    @classmethod
    def from_dict(cls, payload):
        obj = cls("")
        obj.stoi = dict(payload["stoi"])
        obj.itos = {i: ch for ch, i in obj.stoi.items()}
        return obj
