@echo off
REM Fix dependency conflicts for Anomaly Detection

echo ================================================
echo Fixing Dependencies for Anomaly Detection
echo ================================================
echo.

echo Step 1: Uninstalling conflicting packages...
pip uninstall -y numpy torch torchvision scipy opencv-python opencv-python-headless

echo.
echo Step 2: Installing compatible core packages...
pip install "numpy>=2.0.0"
pip install "torch==2.6.0" "torchvision==0.21.0"
pip install "scipy>=1.10.0,<1.17.0"

echo.
echo Step 3: Installing remaining dependencies...
pip install pandas scikit-learn matplotlib seaborn ta tqdm joblib

echo.
echo ================================================
echo Installation Complete!
echo ================================================
echo.
echo You can now run:
echo   python example.py
echo   python main.py --help
echo.

pause

