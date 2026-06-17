#!/usr/bin/env python3
"""
Simple verification script to check file structure and basic syntax
"""

import os
import sys


def check_file_exists(filepath):
    """Check if a file exists and is readable"""
    if os.path.isfile(filepath):
        print(f"✓ {filepath}")
        return True
    else:
        print(f"✗ {filepath} (MISSING)")
        return False


def check_directory_exists(dirpath):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print(f"✓ {dirpath}/")
        return True
    else:
        print(f"✗ {dirpath}/ (MISSING)")
        return False


def main():
    print("Stem+MIDI Pro - File Structure Verification")
    print("=" * 50)

    all_good = True

    # Check root files
    print("\nRoot Files:")
    all_good &= check_file_exists("stem_midi_pro/main.py")
    all_good &= check_file_exists("stem_midi_pro/demo.py")
    all_good &= check_file_exists("stem_midi_pro/Dockerfile")
    all_good &= check_file_exists("stem_midi_pro/requirements.txt")

    # Check configs directory
    print("\nConfiguration:")
    all_good &= check_directory_exists("stem_midi_pro/configs")
    all_good &= check_file_exists("stem_midi_pro/configs/model_config.yaml")

    # Check models directory
    print("\nModels:")
    all_good &= check_directory_exists("stem_midi_pro/models")
    all_good &= check_file_exists("stem_midi_pro/models/mamba_separator.py")
    all_good &= check_file_exists("stem_midi_pro/models/mamba_transcriber.py")
    all_good &= check_file_exists("stem_midi_pro/models/confidence_injector.py")
    all_good &= check_file_exists("stem_midi_pro/models/losses.py")

    # Check utils directory
    print("\nUtilities:")
    all_good &= check_directory_exists("stem_midi_pro/utils")
    all_good &= check_file_exists("stem_midi_pro/utils/quality_gates.py")

    # Check user content
    print("\nUser Content Templates:")
    all_good &= check_directory_exists("stem_midi_pro/user_content")
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
        all_good &= check_file_exists(f"stem_midi_pro/user_content/{filename}")

    print("\n" + "=" * 50)
    if all_good:
        print("✓ ALL FILES PRESENT - Structure is correct!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r stem_midi_pro/requirements.txt")
        print("2. Run demo: python stem_midi_pro/demo.py")
        print("3. For development: modify the component models as needed")
    else:
        print("✗ SOME FILES MISSING - Please check the structure above")
        sys.exit(1)


if __name__ == "__main__":
    main()
