"""
Cleanup Script - Remove old test files and organize project
Run this AFTER your current training completes
"""

import os
import shutil
from pathlib import Path

# Get anomaly_detection directory
script_dir = Path(__file__).parent
anomaly_dir = script_dir

print("="*80)
print("PROJECT CLEANUP")
print("="*80)

# Files to remove (old test files)
files_to_remove = [
    'test_reconstruction_only.py',
    'quick_test.py',
    'debug_scaling.py',
    'example_minimal.py',
    'quickstart.py',
    'train_final_fixed.py',
    'train_improved.py',
    'train_anomaly_detector.py',
    'validate_anomaly_detector.py'
]

# Directories to remove (old outputs)
dirs_to_remove = [
    'example_outputs',
    'quick_test_outputs',
    'final_outputs',
    'validation_outputs'
]

# Markdown files to keep (important docs)
docs_to_keep = [
    'README.md',
    'TRAINING_GUIDE.md',
    'PROJECT_ALIGNMENT.md',
    'GETTING_STARTED.md'
]

# Markdown files to remove (old status files)
docs_to_remove = [
    'FIXES_IMPLEMENTED.md',
    'FIX_SUMMARY.md',
    'NUMPY_FIX.md',
    'RUN_THIS_FOR_BEST_RESULTS.md'
]

print("\n[1/3] Removing old test files...")
removed_files = 0
for filename in files_to_remove:
    filepath = anomaly_dir / filename
    if filepath.exists():
        filepath.unlink()
        print(f"  ✓ Removed: {filename}")
        removed_files += 1

print(f"\nRemoved {removed_files} old test files")

print("\n[2/3] Removing old output directories...")
removed_dirs = 0
for dirname in dirs_to_remove:
    dirpath = anomaly_dir / dirname
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
        print(f"  ✓ Removed: {dirname}/")
        removed_dirs += 1

print(f"\nRemoved {removed_dirs} old directories")

print("\n[3/3] Removing old documentation...")
removed_docs = 0
for filename in docs_to_remove:
    filepath = anomaly_dir / filename
    if filepath.exists():
        filepath.unlink()
        print(f"  ✓ Removed: {filename}")
        removed_docs += 1

print(f"\nRemoved {removed_docs} old docs")

# Rename example.py if it exists
example_file = anomaly_dir / 'example.py'
if example_file.exists():
    new_name = anomaly_dir / 'demo_basic.py'
    example_file.rename(new_name)
    print(f"\n✓ Renamed: example.py → demo_basic.py")

# Rename example_validation.py if it exists
example_val = anomaly_dir / 'example_validation.py'
if example_val.exists():
    new_name = anomaly_dir / 'demo_validation.py'
    example_val.rename(new_name)
    print(f"✓ Renamed: example_validation.py → demo_validation.py")

print("\n" + "="*80)
print("CLEANUP COMPLETE!")
print("="*80)

print("\nFinal Project Structure:")
print("""
anomaly_detection/
├── 📄 MAIN TRAINING FILES
│   ├── train_improved_full.py      ⭐ RECOMMENDED (F1 > 70%)
│   ├── train_working_simple.py     ✓ Basic version (F1 > 60%)
│   └── quick_test_improved.py      🚀 Quick test (20 epochs)
│
├── 📄 DEMO FILES
│   ├── demo_basic.py               Simple demo
│   ├── demo_validation.py          Validation demo
│   └── main.py                     Legacy main
│
├── 📚 DOCUMENTATION
│   ├── README.md                   Overview
│   ├── TRAINING_GUIDE.md          ⭐ How to train
│   ├── PROJECT_ALIGNMENT.md       ⭐ Thesis alignment
│   └── GETTING_STARTED.md         Quick start
│
├── 📁 MODELS & DATA
│   ├── models/                     Model implementations
│   ├── data/                       Data utilities
│   ├── utils/                      Helper functions
│   ├── training/                   Training utilities
│   └── validation/                 Validation utilities
│
├── 📁 OUTPUTS (Created during training)
│   ├── improved_outputs_*/        ⭐ Best results
│   └── working_outputs/           Basic results
│
└── 📄 CONFIG
    ├── requirements.txt            Dependencies
    └── __init__.py                Package init
""")

print("\n🎯 RECOMMENDED NEXT STEPS:")
print("1. Wait for current training to complete")
print("2. Run this cleanup script: python cleanup_project.py")
print("3. Start fresh training: python train_improved_full.py")
print("4. Check results in improved_outputs_*/")
print("\n" + "="*80)

