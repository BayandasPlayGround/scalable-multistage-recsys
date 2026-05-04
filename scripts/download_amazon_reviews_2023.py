from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path


DATASET_PAGE_URL = "https://amazon-reviews-2023.github.io/"
DEFAULT_DESTINATION = Path("amazon_review_data")
DATASET_FILES = {
    "All_Beauty": {
        "review_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz",
        "review_file": "All_Beauty.jsonl",
        "metadata_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz",
        "metadata_file": "meta_All_Beauty.jsonl.gz",
    },
    "Automotive": {
        "review_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Automotive.jsonl.gz",
        "review_file": "Automotive.jsonl",
        "metadata_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Automotive.jsonl.gz",
        "metadata_file": "meta_Automotive.jsonl.gz",
    },
    "Industrial_and_Scientific": {
        "review_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Industrial_and_Scientific.jsonl.gz",
        "review_file": "Industrial_and_Scientific.jsonl",
        "metadata_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Industrial_and_Scientific.jsonl.gz",
        "metadata_file": "meta_Industrial_and_Scientific.jsonl.gz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Amazon Reviews 2023 category files expected by this project.",
        epilog=f"Dataset documentation: {DATASET_PAGE_URL}",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Target data directory. Defaults to amazon_review_data in the current workspace.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(DATASET_FILES),
        default=sorted(DATASET_FILES),
        help="Categories to download.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument(
        "--keep-review-archives",
        action="store_true",
        help="Keep downloaded review .jsonl.gz archives after extracting the .jsonl files.",
    )
    return parser.parse_args()


def download_file(url: str, target: Path, *, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        print(f"exists: {target}")
        return

    temporary_target = target.with_suffix(target.suffix + ".part")
    if temporary_target.exists():
        temporary_target.unlink()

    print(f"download: {url}")
    print(f"target:   {target}")
    try:
        with urllib.request.urlopen(url) as response, open(temporary_target, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary_target.replace(target)
    except Exception:
        if temporary_target.exists():
            temporary_target.unlink()
        raise


def extract_gzip(source: Path, target: Path, *, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        print(f"exists: {target}")
        return

    temporary_target = target.with_suffix(target.suffix + ".part")
    if temporary_target.exists():
        temporary_target.unlink()

    print(f"extract: {source}")
    print(f"target:  {target}")
    try:
        with gzip.open(source, "rb") as compressed, open(temporary_target, "wb") as handle:
            shutil.copyfileobj(compressed, handle, length=1024 * 1024)
        temporary_target.replace(target)
    except Exception:
        if temporary_target.exists():
            temporary_target.unlink()
        raise


def download_category(category: str, destination: Path, *, force: bool, keep_review_archives: bool) -> None:
    config = DATASET_FILES[category]
    review_target = destination / str(config["review_file"])
    review_archive = destination / f"{config['review_file']}.gz"
    metadata_target = destination / "metadata" / str(config["metadata_file"])

    if not review_target.exists() or force:
        download_file(str(config["review_url"]), review_archive, force=force)
        extract_gzip(review_archive, review_target, force=force)
        if not keep_review_archives and review_archive.exists():
            review_archive.unlink()
            print(f"removed: {review_archive}")
    else:
        print(f"exists: {review_target}")

    download_file(str(config["metadata_url"]), metadata_target, force=force)


def main() -> int:
    args = parse_args()
    destination = args.destination.expanduser().resolve()
    print(f"dataset page: {DATASET_PAGE_URL}")
    print(f"destination:  {destination}")
    for category in args.categories:
        print(f"\ncategory: {category}")
        download_category(
            category,
            destination,
            force=bool(args.force),
            keep_review_archives=bool(args.keep_review_archives),
        )
    print("\ncomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
