@echo off
REM YOLO Custom Object Detection - Quick Start Script (Windows)
REM This script helps you get started with the project

echo ==================================
echo YOLO Object Detection Quick Start
echo ==================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Dependencies installed

REM Create necessary directories
echo.
echo Creating project directories...
mkdir dataset\images\train 2>nul
mkdir dataset\images\val 2>nul
mkdir dataset\images\test 2>nul
mkdir dataset\labels\train 2>nul
mkdir dataset\labels\val 2>nul
mkdir dataset\labels\test 2>nul
mkdir raw_data\images 2>nul
mkdir raw_data\labels 2>nul
echo Directories created

REM Show next steps
echo.
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo Next steps:
echo.
echo 1. Prepare your dataset:
echo    - Place images in: raw_data\images\
echo    - Place labels in: raw_data\labels\
echo    - Run: python prepare_data.py
echo.
echo 2. Configure training:
echo    - Edit config.yaml
echo    - Set your class names
echo.
echo 3. Validate dataset:
echo    python train.py --validate-only
echo.
echo 4. Start training:
echo    python train.py
echo.
echo 5. Test your model:
echo    python test.py --model runs\train\yolo_custom\weights\best.pt --source test.jpg
echo.
echo For more information, see README.md
echo For examples, run: python examples.py
echo.
pause
