"""
YOLO Testing and Inference Script
Run object detection inference on images, videos, or webcam feed.
"""
import os
import argparse
from pathlib import Path
from typing import Union, List
import cv2
import numpy as np
from ultralytics import YOLO
import torch


class YOLOInference:
    """YOLO inference handler for object detection."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45):
        """
        Initialize inference handler.
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.load_model()
        
    def load_model(self):
        """Load YOLO model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        print(f"Model loaded successfully on {self.device}")
        
        # Print model info
        print(f"Model classes: {self.model.names}")
        
    def predict_image(self, image_path: str, save_path: str = None,
                     show: bool = False) -> dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            save_path: Path to save output image (optional)
            show: Whether to display the result
            
        Returns:
            Dictionary containing detection results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\nProcessing: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save_path is not None,
            show=show,
            device=self.device
        )
        
        # Extract results
        result = results[0]
        detections = self.parse_results(result)
        
        # Print detections
        self.print_detections(detections, image_path)
        
        # Save annotated image if requested
        if save_path:
            annotated_img = self.draw_detections(image_path, detections)
            cv2.imwrite(save_path, annotated_img)
            print(f"✓ Saved result to: {save_path}")
        
        return detections
    
    def predict_folder(self, folder_path: str, output_folder: str = None,
                      extensions: List[str] = None):
        """
        Run inference on all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            output_folder: Path to save results (optional)
            extensions: List of valid image extensions
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Get all image files
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in extensions]
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return
        
        print(f"\nFound {len(image_files)} images")
        
        # Create output folder if needed
        if output_folder:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Process each image
        all_results = {}
        for img_file in image_files:
            save_path = None
            if output_folder:
                save_path = str(Path(output_folder) / img_file.name)
            
            results = self.predict_image(str(img_file), save_path=save_path)
            all_results[img_file.name] = results
        
        print(f"\n✓ Processed {len(image_files)} images")
        return all_results
    
    def predict_video(self, video_path: str, output_path: str = None,
                     show: bool = False):
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show: Whether to display the result
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\nProcessing video: {video_path}")
        
        # Run inference
        results = self.model.predict(
            source=video_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=output_path is not None,
            show=show,
            device=self.device,
            stream=True  # Use streaming for videos
        )
        
        # Process results
        frame_count = 0
        for result in results:
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames
                detections = self.parse_results(result)
                print(f"Frame {frame_count}: {len(detections['boxes'])} detections")
        
        print(f"✓ Processed {frame_count} frames")
    
    def predict_webcam(self):
        """Run real-time inference on webcam feed."""
        print("\nStarting webcam inference...")
        print("Press 'q' to quit")
        
        # Run inference on webcam
        results = self.model.predict(
            source=0,  # 0 for default webcam
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            show=True,
            device=self.device,
            stream=True
        )
        
        for result in results:
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def parse_results(self, result) -> dict:
        """Parse YOLO results into a structured dictionary."""
        boxes = result.boxes
        
        detections = {
            'boxes': [],
            'confidences': [],
            'class_ids': [],
            'class_names': []
        }
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Extract box coordinates (x1, y1, x2, y2)
                xyxy = box.xyxy[0].cpu().numpy()
                detections['boxes'].append(xyxy)
                
                # Extract confidence
                conf = float(box.conf[0].cpu().numpy())
                detections['confidences'].append(conf)
                
                # Extract class
                cls_id = int(box.cls[0].cpu().numpy())
                detections['class_ids'].append(cls_id)
                detections['class_names'].append(self.model.names[cls_id])
        
        return detections
    
    def print_detections(self, detections: dict, source: str):
        """Print detection results in a readable format."""
        n_detections = len(detections['boxes'])
        
        if n_detections == 0:
            print(f"  No objects detected")
            return
        
        print(f"  Detected {n_detections} object(s):")
        for i in range(n_detections):
            class_name = detections['class_names'][i]
            confidence = detections['confidences'][i]
            box = detections['boxes'][i]
            print(f"    {i+1}. {class_name}: {confidence:.2%} "
                  f"at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
    
    def draw_detections(self, image_path: str, detections: dict) -> np.ndarray:
        """Draw bounding boxes on image."""
        img = cv2.imread(image_path)
        
        for i, box in enumerate(detections['boxes']):
            x1, y1, x2, y2 = map(int, box)
            class_name = detections['class_names'][i]
            confidence = detections['confidences'][i]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2%}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(img, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return img


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on images, videos, or webcam"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLO model (.pt file)'
    )
    parser.add_argument(
        '--source',
        type=str,
        help='Path to image, video, or folder. Use "webcam" for webcam feed'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save output (for images/folders/videos)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results in a window'
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("YOLO Object Detection Inference")
    print(f"{'='*60}\n")
    
    # Initialize inference handler
    try:
        inference = YOLOInference(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run inference based on source type
    try:
        if not args.source:
            print("❌ Error: --source is required")
            print("Usage examples:")
            print("  Image:   python test.py --model best.pt --source image.jpg")
            print("  Folder:  python test.py --model best.pt --source ./images/")
            print("  Video:   python test.py --model best.pt --source video.mp4")
            print("  Webcam:  python test.py --model best.pt --source webcam")
            return
        
        if args.source.lower() == 'webcam':
            inference.predict_webcam()
        elif os.path.isfile(args.source):
            # Check if it's an image or video
            ext = Path(args.source).suffix.lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                inference.predict_video(args.source, args.output, args.show)
            else:
                inference.predict_image(args.source, args.output, args.show)
        elif os.path.isdir(args.source):
            inference.predict_folder(args.source, args.output)
        else:
            print(f"❌ Error: Source not found: {args.source}")
            
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
