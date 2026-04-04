import os
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from ultralytics import YOLO


class YoloCropApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Crop Tool")
        self.root.geometry("760x420")

        self.model_path_var = tk.StringVar(value="runs/train/yolo_custom/weights/best.pt")
        self.image_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value="crops")
        self.conf_var = tk.StringVar(value="0.25")
        self.selected_image_paths = []

        self.model = None

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 8}

        tk.Label(self.root, text="Model (.pt):", anchor="w").grid(row=0, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.model_path_var, width=70).grid(row=0, column=1, sticky="we", **pad)
        tk.Button(self.root, text="Browse", command=self.browse_model, width=12).grid(row=0, column=2, **pad)

        tk.Label(self.root, text="Image(s):", anchor="w").grid(row=1, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.image_path_var, width=70).grid(row=1, column=1, sticky="we", **pad)
        tk.Button(self.root, text="Upload", command=self.browse_image, width=12).grid(row=1, column=2, **pad)

        tk.Label(self.root, text="Save Folder:", anchor="w").grid(row=2, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.output_dir_var, width=70).grid(row=2, column=1, sticky="we", **pad)
        tk.Button(self.root, text="Choose", command=self.browse_output_dir, width=12).grid(row=2, column=2, **pad)

        tk.Label(self.root, text="Confidence (0-1):", anchor="w").grid(row=3, column=0, sticky="w", **pad)
        tk.Entry(self.root, textvariable=self.conf_var, width=12).grid(row=3, column=1, sticky="w", **pad)

        tk.Button(self.root, text="Detect + Crop", command=self.detect_and_crop, width=18, height=2).grid(
            row=4, column=1, sticky="w", **pad
        )

        self.log_text = tk.Text(self.root, height=14, width=92)
        self.log_text.grid(row=5, column=0, columnspan=3, padx=10, pady=(6, 10), sticky="nsew")

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(5, weight=1)

        self.log("Ready. Select model, image(s), and output folder.")

    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if file_path:
            self.model_path_var.set(file_path)
            self.model = None
            self.log(f"Model selected: {file_path}")

    def browse_image(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Image(s)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if file_paths:
            self.selected_image_paths = list(file_paths)
            if len(self.selected_image_paths) == 1:
                self.image_path_var.set(self.selected_image_paths[0])
            else:
                self.image_path_var.set(f"{len(self.selected_image_paths)} images selected")
            self.log(f"Selected {len(self.selected_image_paths)} image(s)")

    def browse_output_dir(self):
        folder_path = filedialog.askdirectory(title="Choose Folder to Save Crops")
        if folder_path:
            self.output_dir_var.set(folder_path)
            self.log(f"Output folder selected: {folder_path}")

    def _load_model(self):
        model_path = self.model_path_var.get().strip()
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError("Invalid model path. Please select a valid .pt file.")

        if self.model is None:
            self.log("Loading model...")
            self.model = YOLO(model_path)
            self.log("Model loaded successfully.")

    def detect_and_crop(self):
        try:
            image_path = self.image_path_var.get().strip()
            output_dir = self.output_dir_var.get().strip()
            conf = float(self.conf_var.get().strip())

            image_paths = []
            if self.selected_image_paths:
                image_paths = [path for path in self.selected_image_paths if os.path.isfile(path)]
            elif image_path and os.path.isfile(image_path):
                image_paths = [image_path]

            if not image_paths:
                messagebox.showerror("Error", "Please upload one or more valid image files.")
                return

            if conf < 0 or conf > 1:
                messagebox.showerror("Error", "Confidence must be between 0 and 1.")
                return

            if not output_dir:
                messagebox.showerror("Error", "Please choose a folder to save crops.")
                return

            self._load_model()

            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            total_saved = 0
            processed_count = 0
            self.log(f"Running detection on {len(image_paths)} image(s)...")

            for current_image_path in image_paths:
                image = cv2.imread(current_image_path)
                if image is None:
                    self.log(f"Skipped unreadable image: {current_image_path}")
                    continue

                results = self.model.predict(source=current_image_path, conf=conf, verbose=False)
                result = results[0]
                boxes = result.boxes
                processed_count += 1

                if boxes is None or len(boxes) == 0:
                    self.log(f"No detections: {current_image_path}")
                    continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_stem = Path(current_image_path).stem
                per_image_saved = 0

                for idx, box in enumerate(boxes, start=1):
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    cls_id = int(box.cls[0].item())
                    conf_score = float(box.conf[0].item())
                    class_name = result.names.get(cls_id, str(cls_id))

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    filename = f"{image_stem}_{timestamp}_{idx}_{class_name}_{conf_score:.2f}.jpg"
                    save_path = output_dir_path / filename
                    cv2.imwrite(str(save_path), crop)
                    per_image_saved += 1
                    total_saved += 1

                self.log(f"Saved {per_image_saved} crop(s) from: {current_image_path}")

            if total_saved == 0:
                self.log("Processing complete. No valid crops were saved.")
                messagebox.showwarning("Done", "No valid crops were saved.")
                return

            done_message = (
                f"Processed {processed_count} image(s).\n"
                f"Saved {total_saved} cropped image(s) to:\n{output_dir_path}"
            )
            self.log(done_message)
            messagebox.showinfo("Done", done_message)

        except ValueError:
            messagebox.showerror("Error", "Confidence must be a numeric value, e.g. 0.25")
        except Exception as exc:
            self.log(f"Error: {exc}")
            messagebox.showerror("Error", str(exc))

    def log(self, text: str):
        self.log_text.insert("end", f"{text}\n")
        self.log_text.see("end")


def main():
    root = tk.Tk()
    app = YoloCropApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
