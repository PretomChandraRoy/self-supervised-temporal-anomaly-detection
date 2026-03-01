#!/bin/bash
# Fix NumPy compatibility issues

echo "================================================"
echo "Fixing NumPy Compatibility for Anomaly Detection"
echo "================================================"
echo ""

echo "Step 1: Uninstalling incompatible NumPy 2.x..."
pip uninstall -y numpy

echo ""
echo "Step 2: Installing compatible NumPy 1.26.x..."
pip install "numpy<2.0.0"

echo ""
echo "Step 3: Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "================================================"
echo "✓ Installation Complete!"
echo "================================================"
echo ""
echo "You can now run:"
echo "  python example.py"
echo "  python main.py --help"
echo ""

