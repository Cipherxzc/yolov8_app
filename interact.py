import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import os
import numpy as np
import cv2
import sys
from ultralytics import YOLO


# 获取模型路径
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")

def load_model(model):
    # 检查是否是绝对路径
    if os.path.isabs(model):
        if os.path.isfile(model):
            return YOLO(model)
        else:
            raise FileNotFoundError(f"Model file '{model}' not found.")
    
    # 检查是否是相对路径
    relative_path = os.path.join(current_dir, model)
    if os.path.isfile(relative_path):
        return YOLO(relative_path)
    
    # 检查是否在 models 目录中
    model_in_models_dir = os.path.join(models_dir, model)
    if os.path.isfile(model_in_models_dir):
        return YOLO(model_in_models_dir)
    
    # 在 models 目录中搜索模型文件
    for root, _, files in os.walk(models_dir):
        if model in files:
            return YOLO(os.path.join(root, model))

    raise FileNotFoundError(f"Model file '{model}' not found in '{models_dir}' or specified path.")

# 初始化模型
model = load_model("lindevs/yolov8n-face-lindevs.pt")

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("800x600")

        self.left_frame = tk.Frame(root, width=200)
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="ns")

        self.upload_btn = tk.Button(self.left_frame, text="上传图片", command=self.upload_photo)
        self.upload_btn.pack(pady=5)

        self.execute_btn = tk.Button(self.left_frame, text="执行", command=self.execute_detection)
        self.execute_btn.pack(pady=5)

        self.right_frame = tk.Frame(root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.horizontal_frame = tk.Frame(self.right_frame)
        self.horizontal_frame.grid(row=0, column=0, sticky="ew")

        self.image_label = tk.Label(self.horizontal_frame, text="No image")
        self.image_label.pack(side="left", padx=5)

        self.result_box = tk.Text(self.right_frame, height=20, width=60)
        self.result_box.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=1)

        self.image_path = None

    def upload_photo(self):
        self.image_path = filedialog.askopenfilename()
        print(f"Selected image path: {self.image_path}")
        if self.image_path:
            img = Image.open(self.image_path)
            img = img.resize((400, 400))
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img, text="")
            self.image_label.image = img

    def execute_detection(self):
        try:
            if not self.image_path:
                messagebox.showerror("Error", "No image uploaded!")
                return

            results = model(self.image_path)

            if isinstance(results, list) and len(results) > 0:
                results = results[0]
                if hasattr(results, 'boxes') and results.boxes is not None:
                    faces = results.boxes.xyxy.cpu().numpy()                    #gpu张量需要放到cpu上转成numpy
                    print(f"Detected faces: {faces}")

                    image = cv2.imread(self.image_path)
                    image_height, image_width = image.shape[:2]
                    face_count = 0

                    # 弹出对话框选择保存文件夹
                    save_dir = filedialog.askdirectory(title="Select Directory to Save Faces")
                    if not save_dir:
                        messagebox.showerror("Error", "No directory selected!")
                        return

                    for face in faces:
                        x1, y1, x2, y2 = map(int, face[:4])
                        
                        # 扩大框到原来的两倍
                        width = x2 - x1
                        height = y2 - y1
                        new_x1 = max(0, x1 - width // 2)
                        new_y1 = max(0, y1 - height // 2)
                        new_x2 = min(image_width, x2 + width // 2)
                        new_y2 = min(image_height, y2 + height // 2)
                        
                        face_image = image[new_y1:new_y2, new_x1:new_x2]
                        face_save_path = os.path.join(save_dir, f"face_{face_count}.jpg")
                        cv2.imwrite(face_save_path, face_image)
                        face_count += 1

                    messagebox.showinfo("Success", f"Detected faces saved as individual images in {save_dir}")
                else:
                    print("No faces detected.")
            else:
                print("Detection failed or no faces detected.")

            self.result_box.delete("1.0", tk.END)
            for face in faces:
                self.result_box.insert(tk.END, f"Face: {face}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
