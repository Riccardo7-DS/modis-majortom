"""
Upload a local Zarr store from data/mtg_data to a HuggingFace dataset repo.

Usage:
    uv run test/upload_hf.py --repo <username>/<repo-name> [--zarr MTG_FCI_mtg_2025_06.zarr]

Requires HF_TOKEN in .env or the environment, or a prior `huggingface-cli login`.
"""
import argparse
from pathlib import Path

from modis_majortom.utils.general import MTG_DATA_PATH, upload_zarr_to_huggingface


def main():
    parser = argparse.ArgumentParser(description="Upload Zarr to HuggingFace")
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. myuser/mtg-fci")
    parser.add_argument(
        "--zarr",
        default="MTG_FCI_mtg_2025_06.zarr",
        help="Name of the Zarr directory inside data/mtg_data (default: MTG_FCI_mtg_2025_06.zarr)",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="HuggingFace repo type (default: dataset)",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel upload threads (default: 4)")
    args = parser.parse_args()

    zarr_path = MTG_DATA_PATH / args.zarr
    if not zarr_path.exists():
        raise FileNotFoundError(
            f"{zarr_path} not found. Run test/download.py first."
        )

    print(f"Uploading {zarr_path} → {args.repo} ({args.repo_type})")
    upload_zarr_to_huggingface(
        local_zarr_path=zarr_path,
        repo_id=args.repo,
        repo_type=args.repo_type,
        num_workers=args.num_workers,
    )
    print("Done.")


if __name__ == "__main__":
    main()
