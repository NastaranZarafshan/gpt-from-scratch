from typing import Iterable, List

class ByteTokenizer:
    """UTF-8 byte tokenizer. IDs 0..255 are bytes; 256..258 are special tokens."""
    PAD, BOS, EOS = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos: ids.insert(0, self.BOS)
        if add_eos: ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        data = []
        for i in ids:
            i = int(i)
            if 0 <= i <= 255: data.append(i)
            elif not skip_special: data.extend(f"<{['PAD','BOS','EOS'][i-256]}>".encode())
        return bytes(data).decode("utf-8", errors="replace")
