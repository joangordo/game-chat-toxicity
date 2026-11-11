"""
00 - Download the data.

  * CONDA (Dota 2 chat): downloaded automatically from GitHub.
  * GameTox (World of Tanks chat): hosted on Google Drive by the shared-task
    organisers. The script tries to download it with `gdown`; if Google blocks
    the automatic download, it tells you what to download by hand.

Run from the project root:
    python scripts/00_download_data.py
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

CONDA_URL = "https://raw.githubusercontent.com/usydnlp/CONDA/main/data/CONDA_{}.csv"
GAMETOX_DRIVE = {
    "train": "https://drive.google.com/drive/folders/1HkfwexOpX1S9gRrMeCFMfZJjsBs0hQRu",
    "val": "https://drive.google.com/drive/folders/1AfQuH7_ixt3qwj3cXbt7Db4prcXZJcAB",
    "test": "https://drive.google.com/file/d/1xeXjylC0bLGPTr1wCyhlATjWHORAqhjO/view",
}
EXPECTED = [f"{s}_index_{k}.csv" for s in config.SPLITS for k in ("text", "label")]


def download_conda() -> None:
    config.RAW_CONDA.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid"):
        dest = config.RAW_CONDA / f"CONDA_{name}.csv"
        if dest.exists():
            print(f"  CONDA {name}: already present")
            continue
        urllib.request.urlretrieve(CONDA_URL.format(name), dest)
        print(f"  CONDA {name}: downloaded")


def gametox_status() -> list[str]:
    found = {p.name for p in config.RAW_GAMETOX.rglob("*.csv")}
    return [f for f in EXPECTED if f not in found]


def download_gametox() -> None:
    config.RAW_GAMETOX.mkdir(parents=True, exist_ok=True)
    if not gametox_status():
        print("  GameTox: already present")
        return
    try:
        import gdown
    except ImportError:
        print("  gdown not installed (pip install gdown); skipping automatic download.")
        return
    for split, url in GAMETOX_DRIVE.items():
        out = config.RAW_GAMETOX / split
        out.mkdir(exist_ok=True)
        try:
            if "folders" in url:
                gdown.download_folder(url, output=str(out), quiet=True)
            else:
                path = gdown.download(url, output=str(out) + "/", quiet=True, fuzzy=True)
                if path and path.endswith(".zip"):
                    with zipfile.ZipFile(path) as z:
                        z.extractall(out)
            print(f"  GameTox {split}: downloaded")
        except Exception as e:  # Google Drive often rate-limits automatic downloads
            print(f"  GameTox {split}: automatic download failed ({type(e).__name__}).")


def main() -> None:
    print("CONDA")
    download_conda()
    print("GameTox")
    download_gametox()
    missing = gametox_status()
    if missing:
        print("\nGameTox files still missing:", ", ".join(missing))
        print("Download them by hand from the Google Drive links in the shared-task repository")
        print("(https://github.com/therealthapa/eeuca-toxicity) and place the CSV files anywhere")
        print(f"inside {config.RAW_GAMETOX}. Subfolders are fine.")
        print("If the test labels are not available, the project still runs: test metrics are skipped.")
    else:
        print("\nAll data present.")


if __name__ == "__main__":
    main()
