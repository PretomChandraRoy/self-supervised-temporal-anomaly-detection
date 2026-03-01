y"""
Quick start script - Run this to get started quickly
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")

    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'ta',
        'tqdm'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install -r requirements.txt")
        return False

    print("✓ All dependencies installed\n")
    return True


def main():
    """Quick start guide"""

    print("="*80)
    print("Self-Supervised Anomaly Detection - Quick Start")
    print("="*80)
    print()

    if not check_dependencies():
        return

    print("📚 Available Commands:")
    print()
    print("1. Run Example (Small demonstration)")
    print("   python anomaly_detection/example.py")
    print()
    print("2. Train on Your Data")
    print("   python anomaly_detection/main.py \\")
    print("       --data_path <path/to/your/data.csv> \\")
    print("       --window_size 60 \\")
    print("       --n_epochs 100 \\")
    print("       --mode train")
    print()
    print("3. Test Pre-trained Model")
    print("   python anomaly_detection/main.py \\")
    print("       --data_path <path/to/test/data.csv> \\")
    print("       --checkpoint outputs/checkpoints/final_model.pt \\")
    print("       --mode test")
    print()
    print("4. Full Pipeline (Train + Test)")
    print("   python anomaly_detection/main.py \\")
    print("       --data_path <path/to/data.csv> \\")
    print("       --mode full")
    print()

    print("📂 Data Format:")
    print("   CSV file with columns: open, high, low, close, volume (optional)")
    print("   Example: forexPredictor/H4_EURUSD_2015.csv")
    print()

    print("📊 Outputs:")
    print("   - outputs/checkpoints/       → Trained models")
    print("   - outputs/visualizations/    → Plots and charts")
    print("   - outputs/anomaly_results.csv → Detection results")
    print()

    print("📖 For detailed documentation, see:")
    print("   anomaly_detection/README.md")
    print()

    print("="*80)

    # Ask if user wants to run example
    try:
        response = input("\nWould you like to run the example now? (y/n): ").strip().lower()
        if response == 'y':
            print("\n🚀 Running example...\n")
            subprocess.run([sys.executable, 'anomaly_detection/example.py'])
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == '__main__':
    main()

