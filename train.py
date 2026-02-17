"""
YOLO Training Script for Custom Object Detection
Train YOLOv8 on your custom dataset for specific class detection.
"""
import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


class YOLOTrainer:
    """Custom YOLO trainer for object detection."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model(self):
        """Initialize YOLO model."""
        model_name = self.config.get('model', 'yolov8n.pt')
        
        print(f"\n{'='*60}")
        print(f"Initializing YOLO model: {model_name}")
        print(f"{'='*60}\n")
        
        # Load pretrained model or create new one
        self.model = YOLO(model_name)
        
        # Print model info
        print(f"Model: {model_name}")
        print(f"Device: {self.get_device()}")
        print(f"Pretrained: {self.config.get('pretrained', True)}")
        
    def get_device(self) -> str:
        """Get training device (cuda or cpu)."""
        device = self.config.get('device', '')
        
        if not device:
            device = '0' if torch.cuda.is_available() else 'cpu'
            
        return device
    
    def validate_dataset(self):
        """Validate dataset exists and is properly formatted."""
        dataset_path = Path(self.config['path'])
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {dataset_path}\n"
                f"Please prepare your dataset using prepare_data.py"
            )
        
        # Check for required directories
        train_imgs = dataset_path / self.config['train']
        val_imgs = dataset_path / self.config['val']
        
        if not train_imgs.exists():
            raise FileNotFoundError(f"Training images not found: {train_imgs}")
        if not val_imgs.exists():
            raise FileNotFoundError(f"Validation images not found: {val_imgs}")
        
        # Count images
        train_count = len(list(train_imgs.glob("*")))
        val_count = len(list(val_imgs.glob("*")))
        
        print(f"\nDataset validation:")
        print(f"  Training images: {train_count}")
        print(f"  Validation images: {val_count}")
        print(f"  Classes: {list(self.config['names'].values())}")
        
        if train_count == 0:
            raise ValueError("No training images found!")
        if val_count == 0:
            raise ValueError("No validation images found!")
            
        return True
    
    def train(self):
        """Train the YOLO model."""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")
        
        # Training parameters
        training_args = {
            'data': self.config_path,
            'epochs': self.config.get('epochs', 100),
            'batch': self.config.get('batch', 16),
            'imgsz': self.config.get('imgsz', 640),
            'device': self.get_device(),
            'patience': self.config.get('patience', 50),
            'save': self.config.get('save', True),
            'save_period': self.config.get('save_period', 10),
            'project': self.config.get('project', 'runs/train'),
            'name': self.config.get('name', 'yolo_custom'),
            'exist_ok': self.config.get('exist_ok', False),
            'pretrained': self.config.get('pretrained', True),
            'optimizer': self.config.get('optimizer', 'SGD'),
            'verbose': self.config.get('verbose', True),
            'lr0': self.config.get('lr0', 0.01),
            'lrf': self.config.get('lrf', 0.01),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'hsv_h': self.config.get('hsv_h', 0.015),
            'hsv_s': self.config.get('hsv_s', 0.7),
            'hsv_v': self.config.get('hsv_v', 0.4),
            'degrees': self.config.get('degrees', 0.0),
            'translate': self.config.get('translate', 0.1),
            'scale': self.config.get('scale', 0.5),
            'shear': self.config.get('shear', 0.0),
            'perspective': self.config.get('perspective', 0.0),
            'flipud': self.config.get('flipud', 0.0),
            'fliplr': self.config.get('fliplr', 0.5),
            'mosaic': self.config.get('mosaic', 1.0),
            'mixup': self.config.get('mixup', 0.0),
        }
        
        # Print training configuration
        print("Training Configuration:")
        for key, value in training_args.items():
            print(f"  {key}: {value}")
        print()
        
        # Start training
        results = self.model.train(**training_args)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}\n")
        
        return results
    
    def save_best_model(self, output_path: str = "best_model.pt"):
        """Save the best trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # The best model is automatically saved by ultralytics
        best_model_path = Path(self.config.get('project', 'runs/train')) / \
                         self.config.get('name', 'yolo_custom') / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            print(f"\n✓ Best model saved at: {best_model_path}")
        else:
            print(f"\n⚠ Warning: Could not find best model at {best_model_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for custom object detection"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate dataset without training'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("YOLO Custom Object Detection Training")
    print(f"{'='*60}\n")
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        print("Please create a config.yaml file or specify --config path")
        return
    
    # Initialize trainer
    trainer = YOLOTrainer(config_path=args.config)
    
    # Validate dataset
    try:
        trainer.validate_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Dataset validation failed: {e}")
        return
    
    if args.validate_only:
        print("\n✓ Dataset validation passed!")
        return
    
    # Setup and train model
    try:
        trainer.setup_model()
        results = trainer.train()
        trainer.save_best_model()
        
        print("\n✓ Training completed successfully!")
        print("\nNext steps:")
        print("  1. Check training results in the 'runs/train' directory")
        print("  2. Use test.py to run inference on new images")
        print("  3. Evaluate model performance on test set")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
