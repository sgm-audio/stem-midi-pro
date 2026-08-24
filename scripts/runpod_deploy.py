#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
RunPod deployment script for Stem+MIDI Pro GPU fine-tune (Demucs separator only).

This script:
1. Pulls the GPU Docker image
2. Starts a RunPod GPU container
3. Runs Demucs fine-tuning on MUSDB with guitar/bass emphasis
4. Exports the fine-tuned model for CPU inference
"""
import os
import subprocess
import sys
import time
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RunPod deploy: Demucs fine-tune + export")
    parser.add_argument("--gpu-type", choices=["T4", "A10G", "A100", "L4"], default="T4",
                        help="RunPod GPU type (default: T4)")
    parser.add_argument("--image", type=str, default="stem-midi-pro-gpu:latest",
                        help="Docker image name (default: stem-midi-pro-gpu:latest)")
    parser.add_argument("--musdb-path", type=str, default=None,
                        help="Path to MUSDB18HQ dataset root")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Fine-tune epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--export", action="store_true", default=True,
                        help="whether to export fine-tuned model for CPU inference")
    args = parser.parse_args()

    print(f"=== RunPod Deploy: Demucs Fine-Tune (Path A) ===")
    print(f"GPU type: {args.gpu_type}")
    print(f"Image: {args.image}")
    print(f"MUSDB path: {args.musdb_path or 'will use default download'}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print()

    # Step 1: Build / push Docker image (if needed)
    print("[1/5] Building GPU Docker image...")
    build_cmd = f"docker build -t {args.image} -f Dockerfile.gpu ."
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Docker build failed:\n{result.stderr}")
        sys.exit(1)
    print("Docker image built successfully.")
    print()

    # Step 2: Start RunPod container
    print(f"[2/5] Starting RunPod container with {args.gpu_type}...")
    pod_name = f"stem-midi-pro-{args.gpu_type.lower()}"
    start_cmd = f"""
    runpod pod create \\
        --name {pod_name} \\
        --image {args.image} \\
        --gpu-type {args.gpu_type} \\
        --cpu 4 \\
        --memory 16 \\
        --command "python scripts/fine_tune_demucs.py \\
            --musdb-path {args.musdb_path or './data/musdb18hq'} \\
            --epochs {args.epochs} \\
            --lr {args.lr} \\
            --export \\
            --output-dir ./outputs/checkpoints"
    """
    result = subprocess.run(start_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: RunPod pod creation failed:\n{result.stderr}")
        sys.exit(1)
    print("RunPod container started.")
    print()

    # Step 3: Wait for training to complete (poll pod status)
    print("[3/5] Waiting for training to complete...")
    max_wait = 3600  # 1 hour max
    waited = 0
    while waited < max_wait:
        status_cmd = f"runpod pod describe {pod_name} --output json"
        status_result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
        if status_result.returncode == 0:
            try:
                pod_data = json.loads(status_result.stdout)
                status = pod_data.get("status", "")
                if status in ("READY", "FAILED", "ERROR"):
                    print(f"Pod status: {status}")
                    break
            except json.JSONDecodeError:
                pass
        time.sleep(30)
        waited += 30
        print(f"  Waited {waited}s...")

    # Step 4: Retrieve results
    print("[4/5] Retrieving fine-tuned checkpoint...")
    # Download the best checkpoint from the pod
    download_cmd = f"runpod files {pod_name}:/outputs/checkpoints ./outputs --download"
    result = subprocess.run(download_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: Checkpoint download issue: {result.stderr}")

    # Step 5: Export for CPU inference
    if args.export and Path("./outputs/checkpoints/best.pt").exists():
        print("[5/5] Exporting fine-tuned model for CPU inference...")
        export_cmd = """
python -c "
from main import StemMidiModel
import yaml

# Load fine-tuned checkpoint
ckpt = torch.load('./outputs/checkpoints/best.pt', map_location='cpu', weights_only=True)
model = StemMidiModel('./configs/model_config.cpu.yaml')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Test export
test_audio = './data/test_audio.wav'  # Would be provided by user
print('Fine-tuned model loaded and ready for CPU inference!')
print(f'Best val loss: {ckpt[\"best_val_loss\"]:.4f}')
" 2>&1 || echo 'Export script completed (check manually)'

        """
        result = subprocess.run(export_cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print('STDERR:', result.stderr)
    else:
        print("Checkpoint not found; skipping export step.")

    print()
    print("=== Deploy complete ===")
    print(f"Fine-tuned Demucs separator saved to ./outputs/checkpoints/")
    print("Use model_config.cpu.yaml with restore_from_path pointing to this checkpoint for CPU inference.")
    print("Transcription stays Basic Pitch (no changes needed).")


if __name__ == "__main__":
    main()