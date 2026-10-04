from pathlib import Path
from urllib.request import urlopen
url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
out=Path("data/input.txt"); out.parent.mkdir(exist_ok=True)
out.write_bytes(urlopen(url).read()); print(f"saved {out} ({out.stat().st_size:,} bytes)")
