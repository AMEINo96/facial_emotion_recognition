"""
FER-2013 Dataset Downloader
----------------------------
Downloads the FER-2013 dataset from Kaggle.
Requires Kaggle API credentials (~/.kaggle/kaggle.json) OR
uses an alternative CSV download method.
"""

import os
import sys
import zipfile

DATASET_DIR = "data"
CSV_FILE    = os.path.join(DATASET_DIR, "fer2013.csv")


def download_via_kaggle():
    """Download using the Kaggle Python API."""
    try:
        import kaggle  # noqa: F401
        os.makedirs(DATASET_DIR, exist_ok=True)
        print("[*] Downloading FER-2013 via Kaggle API …")
        os.system(
            f"kaggle datasets download -d msambare/fer2013 -p {DATASET_DIR} --unzip"
        )
        # The unzipped folder structure is data/train & data/test (image folders)
        if os.path.isdir(os.path.join(DATASET_DIR, "train")):
            print("[✓] Dataset downloaded as image folders (train / test).")
            return "image_folders"
        # Fallback – look for CSV
        candidates = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]
        if candidates:
            os.rename(
                os.path.join(DATASET_DIR, candidates[0]), CSV_FILE
            )
            print("[✓] Dataset downloaded as CSV.")
            return "csv"
    except Exception as e:
        print(f"[!] Kaggle download failed: {e}")
    return None


def download_alternative():
    """
    Alternative: Download the widely-mirrored CSV archive from a public URL.
    Replace the URL below with any valid mirror if needed.
    """
    import urllib.request

    url = (
        "https://www.kaggle.com/datasets/msambare/fer2013/download?datasetVersionNumber=1"
    )
    print("[*] Trying alternative download …")
    print(
        "    Please manually download fer2013.csv from:\n"
        "    https://www.kaggle.com/datasets/msambare/fer2013\n"
        f"    and place it at: {os.path.abspath(CSV_FILE)}"
    )


def check_existing():
    """Return 'csv', 'image_folders', or None depending on what already exists."""
    if os.path.isfile(CSV_FILE):
        return "csv"
    if os.path.isdir(os.path.join(DATASET_DIR, "train")):
        return "image_folders"
    return None


def main():
    existing = check_existing()
    if existing:
        print(f"[✓] Dataset already present (format: {existing}). Skipping download.")
        return existing

    os.makedirs(DATASET_DIR, exist_ok=True)

    result = download_via_kaggle()
    if result:
        return result

    download_alternative()
    return None


if __name__ == "__main__":
    fmt = main()
    if fmt is None:
        print("\n[!] Dataset not found. Please download manually before training.")
        sys.exit(1)
    print(f"[✓] Ready  –  format: {fmt}")
