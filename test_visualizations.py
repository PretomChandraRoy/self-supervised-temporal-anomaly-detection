"""
Quick test to verify visualization and Excel generation works
Uses the existing working_outputs/results.json
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_detailed_excel import generate_detailed_results_excel

def test_excel_generation():
    """Test Excel generation on existing results"""

    results_path = "working_outputs/results.json"
    output_dir = "working_outputs"

    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        print("   Run a training first to generate results.json")
        return False

    print("="*80)
    print("TESTING EXCEL GENERATION")
    print("="*80)
    print(f"Using results from: {results_path}")

    try:
        excel_path = generate_detailed_results_excel(results_path, output_dir)

        if os.path.exists(excel_path):
            file_size = os.path.getsize(excel_path) / 1024  # KB
            print(f"\n✅ SUCCESS!")
            print(f"   Excel file created: {excel_path}")
            print(f"   File size: {file_size:.1f} KB")
            return True
        else:
            print(f"\n❌ FAILED!")
            print(f"   Excel file not found at: {excel_path}")
            return False

    except Exception as e:
        print(f"\n❌ ERROR!")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_imports():
    """Test that visualization dependencies are available"""

    print("\n" + "="*80)
    print("TESTING VISUALIZATION DEPENDENCIES")
    print("="*80)

    dependencies = {
        'matplotlib': None,
        'seaborn': None,
        'numpy': None,
        'pandas': None,
        'openpyxl': None,
        'sklearn': None,
    }

    all_ok = True

    for package in dependencies:
        try:
            if package == 'sklearn':
                import sklearn
                dependencies[package] = sklearn.__version__
            elif package == 'openpyxl':
                import openpyxl
                dependencies[package] = openpyxl.__version__
            else:
                module = __import__(package)
                dependencies[package] = module.__version__
            print(f"  ✓ {package:15} version {dependencies[package]}")
        except ImportError:
            print(f"  ✗ {package:15} NOT INSTALLED")
            all_ok = False
        except AttributeError:
            print(f"  ✓ {package:15} installed (version unknown)")

    if not all_ok:
        print("\n⚠️  Some dependencies missing. Install with:")
        print("   pip install matplotlib seaborn numpy pandas openpyxl scikit-learn")
    else:
        print("\n✅ All dependencies installed!")

    return all_ok


if __name__ == '__main__':
    print("\n" + "="*80)
    print("VISUALIZATION & EXCEL GENERATION TEST")
    print("="*80)

    # Test dependencies
    deps_ok = test_visualization_imports()

    # Test Excel generation if dependencies are OK
    if deps_ok:
        excel_ok = test_excel_generation()

        if excel_ok:
            print("\n" + "="*80)
            print("✅ ALL TESTS PASSED!")
            print("="*80)
            print("\nYou can now:")
            print("  1. Run train_improved_full.py to get full visualizations")
            print("  2. Check working_outputs/DETAILED_RESULTS.xlsx for the Excel report")
            print("  3. Use the generated figures in your thesis")
        else:
            print("\n" + "="*80)
            print("⚠️  Excel generation test failed")
            print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️  Please install missing dependencies first")
        print("="*80)
        print("\nRun: pip install matplotlib seaborn numpy pandas openpyxl scikit-learn")

