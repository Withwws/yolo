# YOLO Custom Object Detection Project

A complete YOLO11-based object detection project for training custom models on specific classes. This project includes data preparation, training, and inference capabilities.

## 🚀 Features

- **Custom Class Training**: Train YOLO models for specific object classes
- **Data Preparation Tools**: Utilities to organize and split your dataset
- **Flexible Training**: Configurable hyperparameters and augmentation settings
- **Multiple Inference Modes**: Test on images, videos, folders, or webcam
- **Easy Configuration**: YAML-based configuration system
- **Pre-trained Models**: Leverage YOLO11 pretrained weights

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Webcam (optional, for real-time detection)

## 🛠️ Installation

1. **Clone or create the project directory:**
   ```bash
   cd yolo
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
yolo/
├── config.yaml           # Training configuration
├── train.py             # Training script
├── test.py              # Inference script
├── prepare_data.py      # Data preparation utilities
├── requirements.txt     # Python dependencies
├── dataset/             # Dataset directory (created during prep)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── runs/                # Training outputs (created during training)
    └── train/
        └── yolo_custom/
            └── weights/
                ├── best.pt
                └── last.pt
```

## 📊 Dataset Preparation

### YOLO Label Format

YOLO uses normalized bounding box coordinates in this format:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer class ID (0, 1, 2, ...)
- `x_center, y_center`: Center coordinates (0.0 to 1.0)
- `width, height`: Box dimensions (0.0 to 1.0)

Example label file (`image1.txt`):
```
0 0.5 0.5 0.3 0.4
0 0.2 0.3 0.15 0.2
```

### Preparing Your Dataset

1. **Organize your raw data:**
   ```
   raw_data/
   ├── images/          # All your images
   │   ├── img001.jpg
   │   ├── img002.jpg
   │   └── ...
   └── labels/          # Corresponding YOLO labels
       ├── img001.txt
       ├── img002.txt
       └── ...
   ```

2. **Use the data preparation script:**
   ```python
   from prepare_data import YOLODatasetPreparer
   
   preparer = YOLODatasetPreparer(
       source_dir="./raw_data",
       output_dir="./dataset"
   )
   
   # Create directory structure
   preparer.create_directory_structure()
   
   # Split dataset (70% train, 20% val, 10% test)
   preparer.split_dataset(
       images_dir="./raw_data/images",
       labels_dir="./raw_data/labels",
       train_ratio=0.7,
       val_ratio=0.2,
       test_ratio=0.1
   )
   ```

3. **Or run from command line:**
   ```bash
   python prepare_data.py
   ```

## ⚙️ Configuration

Edit `config.yaml` to customize your training:

```yaml
# Dataset paths
path: ./dataset
train: images/train
val: images/val
test: images/test

# Your custom classes
names:
  0: person  # Change to your target class
  # Add more classes as needed:
  # 1: car
  # 2: bicycle

# Training settings
epochs: 100
batch: 16
imgsz: 640
model: yolo11n.pt  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

# Optimization
lr0: 0.01
optimizer: SGD
```

### Model Size Options

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolo11n | Nano | Fastest | Good |
| yolo11s | Small | Fast | Better |
| yolo11m | Medium | Moderate | Great |
| yolo11l | Large | Slow | Excellent |
| yolo11x | Extra Large | Slowest | Best |

## 🎯 Training

### Basic Training

```bash
python train.py
```

### Validate Dataset Only

```bash
python train.py --validate-only
```

### Custom Configuration

```bash
python train.py --config custom_config.yaml
```

### What Happens During Training

1. Dataset validation
2. Model initialization (downloads pretrained weights if needed)
3. Training loop with progress display
4. Automatic checkpoint saving
5. Best model selection based on validation metrics

### Training Outputs

Results are saved in `runs/train/yolo_custom/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- `results.png` - Training metrics plots
- `confusion_matrix.png` - Confusion matrix
- `val_batch*.jpg` - Validation predictions

## 🧪 Testing and Inference

### Test on a Single Image

```bash
python test.py --model runs/train/yolo_custom/weights/best.pt --source image.jpg --output result.jpg
```

### Test on Multiple Images (Folder)

```bash
python test.py --model runs/train/yolo_custom/weights/best.pt --source ./test_images/ --output ./results/
```

### Test on Video

```bash
python test.py --model runs/train/yolo_custom/weights/best.pt --source video.mp4 --output output.mp4
```

### Real-time Webcam Detection

```bash
python test.py --model runs/train/yolo_custom/weights/best.pt --source webcam
```

### Advanced Options

```bash
python test.py \
    --model best.pt \
    --source image.jpg \
    --output result.jpg \
    --conf 0.5 \        # Confidence threshold (default: 0.25)
    --iou 0.45 \        # IoU threshold for NMS (default: 0.45)
    --show              # Display results in window
```

## 📈 Monitoring Training

### Using TensorBoard (optional)

The ultralytics package supports TensorBoard logging:

```bash
tensorboard --logdir runs/train
```

Then open http://localhost:6006 in your browser.

## 🎓 Example Workflows

### Training a Person Detector

1. **Prepare dataset with person annotations**

2. **Update config.yaml:**
   ```yaml
   names:
     0: person
   epochs: 100
   batch: 16
   ```

3. **Train:**
   ```bash
   python train.py
   ```

4. **Test:**
   ```bash
   python test.py --model runs/train/yolo_custom/weights/best.pt --source test.jpg
   ```

### Fine-tuning for a New Class

1. Start with a smaller model for faster iteration:
   ```yaml
   model: yolo11n.pt
   epochs: 50
   ```

2. Train and evaluate

3. If needed, switch to a larger model:
   ```yaml
   model: yolo11s.pt
   epochs: 100
   ```

## 🔧 Troubleshooting

### Out of Memory Error

- Reduce batch size in `config.yaml`:
  ```yaml
  batch: 8  # or 4
  ```
- Use a smaller model: `yolo11n.pt`

### Low Accuracy

- Increase training epochs
- Add more training data
- Adjust data augmentation parameters
- Try a larger model (yolo11s, yolo11m)

### No Objects Detected

- Lower confidence threshold: `--conf 0.1`
- Check if test images are similar to training images
- Verify model is trained properly (check training metrics)

## 📚 Additional Resources

- [YOLO11 Documentation](https://docs.ultralytics.com/)
- [YOLO Label Tools](https://github.com/heartexlabs/labelImg)
- [Data Augmentation Guide](https://docs.ultralytics.com/modes/train/#augmentation)

## 💡 Tips for Better Results

1. **Quality over Quantity**: 100 well-annotated images > 1000 poor annotations
2. **Balanced Dataset**: Try to have similar numbers of each class
3. **Diverse Data**: Include various lighting, angles, and backgrounds
4. **Regular Validation**: Monitor validation metrics during training
5. **Data Augmentation**: Use augmentation for small datasets

## 📝 License

This project uses the YOLO11 model from Ultralytics, which is licensed under AGPL-3.0.

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📧 Support

For issues with:
- This project: Check the troubleshooting section
- YOLO11: Visit [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- General YOLO questions: Check [Ultralytics Docs](https://docs.ultralytics.com/)

---

**Happy Training! 🎉**
