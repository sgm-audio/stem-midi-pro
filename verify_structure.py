#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""
Simple verification script to check file structure and basic syntax
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OK = "[OK]"
MISSING = "[MISSING]"


def check_file_exists(filepath):
    """Check if a file exists and is readable"""
    if os.path.isfile(filepath):
        print(f"{OK} {filepath}")
        return True
    else:
        print(f"{MISSING} {filepath}")
        return False


def check_directory_exists(dirpath):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print(f"{OK} {dirpath}/")
        return True
    else:
        print(f"{MISSING} {dirpath}/")
        return False


def main():
    print("Stem+MIDI Pro - File Structure Verification")
    print("=" * 50)

    all_good = True

    print("\nRoot Files:")
    all_good &= check_file_exists("main.py")
    all_good &= check_file_exists("api.py")
    all_good &= check_file_exists("demo.py")
    all_good &= check_file_exists("train.py")
    all_good &= check_file_exists("Dockerfile")
    all_good &= check_file_exists("requirements.txt")

    print("\nConfiguration:")
    all_good &= check_directory_exists("configs")
    all_good &= check_file_exists("configs/model_config.yaml")
    all_good &= check_file_exists("example_data_config.yaml")

    print("\nModels:")
    all_good &= check_directory_exists("models")
    all_good &= check_file_exists("models/mamba_separator.py")
    all_good &= check_file_exists("models/mamba_transcriber.py")
    all_good &= check_file_exists("models/confidence_injector.py")
    all_good &= check_file_exists("models/losses.py")

    print("\nUtilities:")
    all_good &= check_directory_exists("utils")
    all_good &= check_file_exists("utils/quality_gates.py")
    all_good &= check_file_exists("utils/template_engine.py")

    print("\nData:")
    all_good &= check_directory_exists("data")
    all_good &= check_file_exists("data/datasets.py")

    print("\nUser Content Templates:")
    all_good &= check_directory_exists("user_content")
    user_content_files = [
        "upload_confirmation.md",
        "progress_updates.md",
        "completion_delivery.md",
        "rights_usage_prompt.md",
        "feedback_refinement.md",
        "implicit_feedback.md",
        "landing_page.md",
    ]
    for filename in user_content_files:
        all_good &= check_file_exists(f"user_content/{filename}")

    print("\nPackage Structure (__init__.py):")
    for pkg_dir in ["models", "utils", "data", "user_content"]:
        all_good &= check_file_exists(f"{pkg_dir}/__init__.py")

    print("\nResearch (informational, not required for v1):")
    all_good &= check_directory_exists("research")
    all_good &= check_directory_exists("research/mamba3_per_track")
    all_good &= check_file_exists("research/mamba3_per_track/per_track_processor.py")
    all_good &= check_file_exists("research/mamba-ssm-reference/mamba/README.md")

    print("\n" + "=" * 50)
    if all_good:
        print(f"{OK} ALL FILES PRESENT - Structure is correct!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run demo: python demo.py")
        print("3. For development: modify the component models as needed")
    else:
        print(f"{MISSING} SOME FILES MISSING - Please check the structure above")
        sys.exit(1)


if __name__ == "__main__":
    main()
