"""Download the Intel Berkeley Research Lab files into data/raw and check them.

    python scripts/download_data.py

The three files are published by MIT CSAIL at
https://db.csail.mit.edu/labdata/labdata.html. If the download is blocked on
your network, save the files by hand into data/raw/ (keep the names below);
the SHA-256 check then tells you whether you have the same bytes we used.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://db.csail.mit.edu/labdata/"
FILES = {
    "data.txt.gz": "d99288c8f406ca6604d359ceaa0d8adfffa79e7095061a1e27dc4399f48c7225",
    "mote_locs.txt": "3865c0263110c24c40e3377690cecaa552e0575cf56cdb9f5f8bd17130b6bf04",
    "connectivity.txt": "267a2bf3fcfb911df7e6746a9aee38bfbdd124caa566e05d2466d8a11666a6d8",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, digest in FILES.items():
        dest = RAW / name
        if not dest.exists():
            print(f"downloading {name} ...")
            try:
                urllib.request.urlretrieve(BASE + name, dest)
            except OSError as exc:
                print(f"  could not download {name}: {exc}. Save it by hand into {RAW}.")
                ok = False
                continue
        got = sha256(dest)
        same = got == digest
        ok &= same
        print(f"{name}: {'OK' if same else 'CHECKSUM MISMATCH ' + got}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
