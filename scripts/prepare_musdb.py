#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Data preparation script for MUSDB18HQ dataset (Phase 2 Path A).

Downloads and extracts MUSDB18HQ dataset for Demucs fine-tuning.
Compatible with the example_data_config.yaml dataset_type: musdb18hq.
"""
import argparse
import os
import shutil
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
    parser.add_argument("--downloader", type=str, default="auto",
                        choices=["auto", "aria2c", "wget", "curl"],
                        help="Downloader to use (default: auto = aria2c -> wget -> curl)")
    parser.add_argument("--aria2c-path", type=str, default="aria2c",
                        help="Path to aria2c binary (default: aria2c on PATH)")
    parser.add_argument("--connections", type=int, default=8,
                        help="aria2c connections per server (default: 8)")
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

    # Download: aria2c first (-c resumes partials), then wget -c, then curl -C -.
    # No subprocess timeout - Ctrl-C and re-run; every downloader below resumes.
    # Canonical source: MUSDB18-HQ, Zenodo record 3338373 (doi:10.5281/zenodo.3338373).
    download_url = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
    tar_path = output_dir / "musdb18hq.zip"
    if tar_path.exists() and tar_path.stat().st_size < 1_000_000:
        print(f"Removing stub {tar_path} (error page, not the dataset) - starting fresh...")
        tar_path.unlink()
    if tar_path.exists():
        print(f"Found partial {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB) - resuming...")

    def _run(cmd):
        print(f"  $ {' '.join(cmd)}")
        try:
            return subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print(f"  SKIP: {cmd[0]} not found")
            return None

    def _dl_aria2c():
        if args.aria2c_path == "aria2c" and shutil.which("aria2c") is None:
            print("  SKIP: aria2c not found")
            return None
        return _run([args.aria2c_path, "-c", "-x", str(args.connections),
                     "--summary-interval=60", "--connect-timeout=30",
                     "--max-tries=5", "--retry-wait=5",
                     "-d", str(output_dir), "-o", tar_path.name, download_url])

    def _dl_wget():
        return _run(["wget", "-c", "--timeout=30", "--tries=5",
                     "-O", str(tar_path), download_url])

    def _dl_curl():
        return _run(["curl", "-L", "-C", "-", "-o", str(tar_path),
                     "--retry", "5", "--connect-timeout", "30", download_url])

    order = {"auto": ("aria2c", "wget", "curl"), "aria2c": ("aria2c",),
             "wget": ("wget",), "curl": ("curl",)}[args.downloader]
    downloaders = {"aria2c": _dl_aria2c, "wget": _dl_wget, "curl": _dl_curl}
    downloaded = False
    for name in order:
        result = downloaders[name]()
        if result is not None and result.returncode == 0 and tar_path.exists():
            downloaded = True
            break
        print(f"{name} failed, trying next...")
    if not downloaded:
        print("ERROR: All downloaders failed.")
        print()
        print("Manual download required:")
        print(f"  {download_url}")
        print("Save to ./data/musdb18hq.zip and re-run with --force")
        sys.exit(1)

    print("Download complete, extracting...")

    # Extract: tar -xf is fast where it handles zip (bsdtar/Windows);
    # stdlib unpack_archive covers the rest (.zip and .tar.gz alike).
    extracted_ok = False
    result = _run(["tar", "-xf", str(tar_path), "-C", str(output_dir)])
    if result is not None and result.returncode == 0:
        extracted_ok = True
    else:
        print("tar failed, trying stdlib unpack...")
        try:
            shutil.unpack_archive(str(tar_path), str(output_dir))
            extracted_ok = True
        except Exception as e:
            print(f"ERROR: Extraction failed: {e}")
    if not extracted_ok:
        print(f"Archive: {tar_path}")
        sys.exit(1)

    # Normalize layout to dest_dir/{train,test,...}. The zip either wraps
    # everything in a top-level folder or extracts train/test at top level.
    # Drive roots hold unreadable system entries - skip those and anything
    # else the OS will not let us stat or move.
    _SYSTEM_NAMES = {"System Volume Information", "$RECYCLE.BIN"}

    def _is_movable_dir(p):
        if p == dest_dir or p.name in _SYSTEM_NAMES or p.name.startswith("$"):
            return False
        try:
            return p.is_dir()
        except OSError:
            return False

    candidates = [d for d in output_dir.iterdir() if _is_movable_dir(d)]
    wrapper = next((d for d in candidates if d.name.startswith("musdb18hq")), None)
    if wrapper is not None:
        print(f"Moving extracted data to {dest_dir}...")
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(wrapper), str(dest_dir))
    elif (output_dir / "train").is_dir():
        print(f"Gathering extracted splits into {dest_dir}...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        for d in candidates:
            target = dest_dir / d.name
            if target.exists():
                continue  # resuming a previous gather
            try:
                shutil.move(str(d), str(target))
            except OSError as e:
                print(f"  SKIP {d.name}: {e}")
    else:
        print("ERROR: Could not find extracted MUSDB18HQ directory")
        print("Contents of data directory:")
        for item in output_dir.iterdir():
            print(f"  {item.name}")
        sys.exit(1)

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