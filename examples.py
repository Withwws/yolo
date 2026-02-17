"""
Example usage of the YOLO training and inference pipeline.
Run this script to see how to use the project programmatically.
"""
from prepare_data import YOLODatasetPreparer, create_label_file
from pathlib import Path


def example_data_preparation():
    """Example: Prepare dataset from raw data."""
    print("=" * 60)
    print("Example 1: Data Preparation")
    print("=" * 60)
    
    # Initialize preparer
    preparer = YOLODatasetPreparer(
        source_dir="./raw_data",
        output_dir="./dataset"
    )
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    preparer.create_directory_structure()
    
    # Split dataset
    # Uncomment when you have data:
    # print("\n2. Splitting dataset...")
    # preparer.split_dataset(
    #     images_dir="./raw_data/images",
    #     labels_dir="./raw_data/labels",
    #     train_ratio=0.7,
    #     val_ratio=0.2,
    #     test_ratio=0.1,
    #     seed=42
    # )
    
    # Validate
    # print("\n3. Validating dataset...")
    # train_count, val_count, test_count = preparer.validate_dataset()
    # print(f"✓ Dataset ready: {train_count} train, {val_count} val, {test_count} test")


def example_create_label():
    """Example: Create a YOLO format label file."""
    print("\n" + "=" * 60)
    print("Example 2: Create Label File")
    print("=" * 60)
    
    # Example: Image with 2 person detections
    # Assuming image is 1920x1080 pixels
    # Format: (class_id, (x_min, y_min, x_max, y_max))
    
    bboxes = [
        (0, (100, 200, 300, 500)),  # First person
        (0, (800, 150, 1000, 550)),  # Second person
    ]
    
    # This would create the label file
    # Uncomment when you have an actual image:
    # create_label_file(
    #     image_path="path/to/image.jpg",
    #     bboxes=bboxes,
    #     output_path="path/to/label.txt"
    # )
    
    print("\nLabel file format:")
    print("class_id x_center y_center width height")
    print("\nExample label content:")
    print("0 0.104167 0.324074 0.104167 0.277778")
    print("0 0.468750 0.324074 0.104167 0.370370")


def example_training_config():
    """Example: Show training configuration options."""
    print("\n" + "=" * 60)
    print("Example 3: Training Configuration")
    print("=" * 60)
    
    print("\nBasic training command:")
    print("  python train.py")
    
    print("\nWith custom config:")
    print("  python train.py --config custom_config.yaml")
    
    print("\nValidate dataset only:")
    print("  python train.py --validate-only")
    
    print("\nKey config parameters:")
    config_example = """
    # In config.yaml:
    epochs: 100        # Number of training epochs
    batch: 16          # Batch size (reduce if OOM)
    imgsz: 640         # Input image size
    model: yolov8n.pt  # Model size (n/s/m/l/x)
    lr0: 0.01          # Initial learning rate
    patience: 50       # Early stopping patience
    """
    print(config_example)


def example_inference():
    """Example: Show inference usage."""
    print("\n" + "=" * 60)
    print("Example 4: Inference Usage")
    print("=" * 60)
    
    print("\nSingle image:")
    print("  python test.py --model best.pt --source image.jpg --output result.jpg")
    
    print("\nFolder of images:")
    print("  python test.py --model best.pt --source ./images/ --output ./results/")
    
    print("\nVideo file:")
    print("  python test.py --model best.pt --source video.mp4 --output output.mp4")
    
    print("\nWebcam (real-time):")
    print("  python test.py --model best.pt --source webcam")
    
    print("\nWith custom thresholds:")
    print("  python test.py --model best.pt --source image.jpg --conf 0.5 --iou 0.45")


def example_programmatic_inference():
    """Example: Using inference programmatically."""
    print("\n" + "=" * 60)
    print("Example 5: Programmatic Inference")
    print("=" * 60)
    
    code_example = """
from test import YOLOInference

# Initialize
inference = YOLOInference(
    model_path="runs/train/yolo_custom/weights/best.pt",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Predict single image
results = inference.predict_image(
    image_path="test.jpg",
    save_path="result.jpg",
    show=True
)

# Access results
for i, detection in enumerate(results['boxes']):
    class_name = results['class_names'][i]
    confidence = results['confidences'][i]
    print(f"Detected {class_name} with {confidence:.2%} confidence")

# Predict folder
all_results = inference.predict_folder(
    folder_path="./test_images",
    output_folder="./results"
)
    """
    print(code_example)


def example_best_practices():
    """Example: Best practices for training."""
    print("\n" + "=" * 60)
    print("Example 6: Best Practices")
    print("=" * 60)
    
    practices = """
1. Dataset Quality:
   - At least 100-500 images per class
   - Diverse lighting and backgrounds
   - Consistent annotation quality
   - Balanced class distribution

2. Starting Small:
   - Begin with yolov8n.pt for fast iteration
   - Use small batch size to test pipeline
   - Validate dataset before full training

3. Hyperparameter Tuning:
   - Start with default settings
   - Adjust batch size based on GPU memory
   - Increase epochs if underfitting
   - Add augmentation for small datasets

4. Monitoring:
   - Watch validation loss (should decrease)
   - Check validation predictions
   - Look for overfitting (train >> val performance)
   - Use early stopping (patience parameter)

5. Inference Optimization:
   - Lower conf threshold if missing objects
   - Higher conf threshold if too many false positives
   - Adjust IoU threshold for overlapping objects
    """
    print(practices)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("YOLO CUSTOM OBJECT DETECTION - EXAMPLES")
    print("=" * 60)
    
    example_data_preparation()
    example_create_label()
    example_training_config()
    example_inference()
    example_programmatic_inference()
    example_best_practices()
    
    print("\n" + "=" * 60)
    print("Quick Start Checklist:")
    print("=" * 60)
    print("""
    □ 1. Prepare your dataset (images + YOLO labels)
    □ 2. Run: python prepare_data.py
    □ 3. Edit config.yaml (set your class names)
    □ 4. Run: python train.py --validate-only
    □ 5. Run: python train.py
    □ 6. Run: python test.py --model best.pt --source test.jpg
    
    See README.md for detailed instructions!
    """)


if __name__ == "__main__":
    main()
