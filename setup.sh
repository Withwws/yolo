#!/bin/bash

# YOLO Custom Object Detection - Quick Start Script
# This script helps you get started with the project

echo "=================================="
echo "YOLO Object Detection Quick Start"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p dataset/images/{train,val,test}
mkdir -p dataset/labels/{train,val,test}
mkdir -p raw_data/images
mkdir -p raw_data/labels
echo "✓ Directories created"

# Show next steps
echo ""
echo "=================================="
echo "✓ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Prepare your dataset:"
echo "   - Place images in: raw_data/images/"
echo "   - Place labels in: raw_data/labels/"
echo "   - Run: python prepare_data.py"
echo ""
echo "2. Configure training:"
echo "   - Edit config.yaml"
echo "   - Set your class names"
echo ""
echo "3. Validate dataset:"
echo "   python train.py --validate-only"
echo ""
echo "4. Start training:"
echo "   python train.py"
echo ""
echo "5. Test your model:"
echo "   python test.py --model runs/train/yolo_custom/weights/best.pt --source test.jpg"
echo ""
echo "For more information, see README.md"
echo "For examples, run: python examples.py"
echo ""
