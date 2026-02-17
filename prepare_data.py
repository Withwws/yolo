"""
Data Preparation Utilities for YOLO Training
This script helps organize and split your dataset for YOLO training.
"""
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple


class YOLODatasetPreparer:
    """Prepare dataset for YOLO training."""
    
    def __init__(self, source_dir: str, output_dir: str = "./dataset"):
        """
        Initialize dataset preparer.
        
        Args:
            source_dir: Directory containing images and labels
            output_dir: Output directory for organized dataset
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
    def create_directory_structure(self):
        """Create YOLO dataset directory structure."""
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
            self.output_dir / "labels" / "test",
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✓ Created directory structure at {self.output_dir}")
        
    def split_dataset(self, 
                     images_dir: str,
                     labels_dir: str,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.2,
                     test_ratio: float = 0.1,
                     seed: int = 42):
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        random.seed(seed)
        
        # Get all image files
        images_path = Path(images_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in images_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to respective directories
        self._copy_files(train_files, "train", labels_dir)
        self._copy_files(val_files, "val", labels_dir)
        self._copy_files(test_files, "test", labels_dir)
        
        print(f"\n✓ Dataset split complete:")
        print(f"  Train: {len(train_files)} images")
        print(f"  Val:   {len(val_files)} images")
        print(f"  Test:  {len(test_files)} images")
        
    def _copy_files(self, image_files: List[Path], split: str, labels_dir: str):
        """Copy image and label files to destination."""
        labels_path = Path(labels_dir)
        
        for img_file in image_files:
            # Copy image
            dst_img = self.output_dir / "images" / split / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Copy corresponding label file
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = self.output_dir / "labels" / split / label_file.name
                shutil.copy2(label_file, dst_label)
            else:
                print(f"⚠ Warning: Label not found for {img_file.name}")
    
    def validate_dataset(self) -> Tuple[int, int, int]:
        """
        Validate dataset structure and return counts.
        
        Returns:
            Tuple of (train_count, val_count, test_count)
        """
        counts = {}
        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / "images" / split
            label_dir = self.output_dir / "labels" / split
            
            img_files = list(img_dir.glob("*"))
            label_files = list(label_dir.glob("*.txt"))
            
            counts[split] = len(img_files)
            
            if len(img_files) != len(label_files):
                print(f"⚠ Warning: {split} set has {len(img_files)} images "
                      f"but {len(label_files)} labels")
        
        return counts['train'], counts['val'], counts['test']


def convert_bbox_to_yolo(bbox: Tuple[int, int, int, int],
                         img_width: int,
                         img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (x_min, y_min, x_max, y_max) to YOLO format.
    
    Args:
        bbox: Bounding box in (x_min, y_min, x_max, y_max) format
        img_width: Image width
        img_height: Image height
        
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center coordinates
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    
    # Calculate width and height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return x_center, y_center, width, height


def create_label_file(image_path: str,
                     bboxes: List[Tuple[int, Tuple[int, int, int, int]]],
                     output_path: str):
    """
    Create a YOLO format label file.
    
    Args:
        image_path: Path to the image
        bboxes: List of (class_id, bbox) tuples
        output_path: Path to save the label file
    """
    from PIL import Image
    
    # Get image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    with open(output_path, 'w') as f:
        for class_id, bbox in bboxes:
            # Convert to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox, img_width, img_height
            )
            
            # Write to file
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} "
                   f"{width:.6f} {height:.6f}\n")


if __name__ == "__main__":
    # Example usage
    print("YOLO Dataset Preparer")
    print("=" * 50)
    
    # Create preparer instance
    preparer = YOLODatasetPreparer(
        source_dir="./raw_data",
        output_dir="./dataset"
    )
    
    # Create directory structure
    preparer.create_directory_structure()
    
    # Example: Split dataset
    # Uncomment and modify paths as needed
    preparer.split_dataset(
        images_dir="./raw_data/images",
        labels_dir="./raw_data/labels",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    
    print("\nTo use this script:")
    print("1. Place your images in a source directory")
    print("2. Place corresponding YOLO labels in a labels directory")
    print("3. Uncomment and modify the split_dataset() call above")
    print("4. Run: python prepare_data.py")
