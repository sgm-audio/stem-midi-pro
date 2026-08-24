#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Data preparation script for MUSDB18HQ dataset (Phase 2 Path A).

Downloads and extracts MUSDB18HQ dataset for Demucs fine-tuning.
Compatible with the example_data_config.yaml dataset_type: musdb18hq.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare MUSDB18HQ dataset for Demucs fine-tune")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory for dataset (default: ./data)")
    parser.add_argument("--dest", type=str, default="musdb18hq",
                        help="Destination subdirectory name")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download if dataset already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dest_dir = output_dir / args.dest

    print("=== MUSDB18HQ Dataset Preparation (Phase 2 Path A) ===")
    print(f"Output directory: {output_dir}")
    print()

    # Check if dataset already exists
    if dest_dir.exists() and not args.force:
        print(f"Dataset already exists at {dest_dir}")
        print("Use --force to re-download.")
        # Verify it has the expected structure
        train_dir = dest_dir / "train"
        if train_dir.exists():
            tracks = len(list(train_dir.glob("*/")))
            print(f"  Found {tracks} training tracks")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MUSDB18HQ dataset...")
    print("(This may take several minutes depending on network speed)")

    # Try using the musdb Python package
    try:
        import musdb
        print("musdb package found, using to download...")
        # musdb does have a download function, but let's use a simpler approach
        print("musdb API available but will use direct download instead")
    except ImportError:
        print("musdb not installed, using direct download approach")

    # Download using wget - MUSDB18HQ is publicly available
    # The dataset is at https://zenodo.org/record/1204852
    # We'll download the tar.gz and extract it
    download_url = "https://zenodo.org/record/1204852/files/musdb18hq.tar.gz?download=1"
    tar_path = output_dir / "musdb18hq.tar.gz"

    try:
        result = subprocess.run(
            ["wget", "--timeout=300", "--tries=3", "-O", str(tar_path), download_url],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            # Try with curl as fallback
            print("wget failed, trying curl...")
            result = subprocess.run(
                ["curl", "-L", "-o", str(tar_path), "--max-time", "300", download_url],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                print(f"ERROR: Download failed:\nwget: {result.stderr}\ncurl: check manual download")
                print()
                print("Manual download required:")
                print(f"  {download_url}")
                print("Save to ./data/musdb18hq.tar.gz and re-run with --force")
                sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Download timed out after 10 minutes")
        print("Please try again or download manually from:")
        print(f"  {download_url}")
        sys.exit(1)

    print("Download complete, extracting...")

    # Extract
    try:
        result = subprocess.run(
            ["tar", "xzf", str(tar_path), "-C", str(output_dir)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"ERROR: Extraction failed: {result.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: tar not found, please install tar or extract manually")
        print(f"Archive: {tar_path}")
        print("Manual extraction required:")
        print(f"  tar -xzf {tar_path} -C {output_dir}/")
        sys.exit(1)

    # Find the extracted directory - MUSDB18HQ structure varies
    extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("musdb18hq")]
    if not extracted_dirs:
        # Maybe the tar stripped the prefix
        extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    if not extracted_dirs:
        print("ERROR: Could not find extracted MUSDB18HQ directory")
        print("Contents of data directory:")
        for item in output_dir.iterdir():
            print(f"  {item.name}")
        sys.exit(1)

    extracted = extracted_dirs[0]

    # Move/rename to expected location
    if extracted != dest_dir:
        print(f"Moving extracted data to {dest_dir}...")
        if dest_dir.exists():
            import shutil
            shutil.rmtree(dest_dir)
        import shutil
        shutil.move(str(extracted), str(dest_dir))

    # Verify structure
    train_dir = dest_dir / "train"
    if train_dir.exists():
        tracks = len(list(train_dir.glob("*/")))
        print(f"MUSDB18HQ dataset prepared successfully!")
        print(f"  Location: {dest_dir}")
        print(f"  Training tracks: {tracks}")
        print()
        print("Next step: Run fine-tune")
        print(f"  python scripts/fine_tune_demucs.py --musdb-path {dest_dir} --epochs 50")
    else:
        print(f"WARNING: Expected 'train' subdirectory not found at {dest_dir}")
        print("Contents:")
        for item in dest_dir.iterdir():
            print(f"  {item.name}")


if __name__ == "__main__":
    main()