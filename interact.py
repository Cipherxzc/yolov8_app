import tkinter as tk
from tkinter import filedialog, messagebox, Menu, Toplevel, Listbox, Button
from PIL import Image, ImageTk
import torch
import os
import cv2
from ultralytics import YOLO

# 获取模型路径
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")

def load_model(model_name):
    # 检查是否是绝对路径
    if os.path.isabs(model_name):
        if os.path.isfile(model_name):
            return YOLO(model_name)
        else:
            raise FileNotFoundError(f"Model file '{model_name}' not found.")
    
    # 检查是否是相对路径
    relative_path = os.path.join(current_dir, model_name)
    if os.path.isfile(relative_path):
        return YOLO(relative_path)
    
    # 检查是否在 models 目录中
    model_in_models_dir = os.path.join(models_dir, model_name)
    if os.path.isfile(model_in_models_dir):
        return YOLO(model_in_models_dir)
    
    # 在 models 目录中搜索模型文件
    for root, _, files in os.walk(models_dir):
        if model_name in files:
            return YOLO(os.path.join(root, model_name))

    raise FileNotFoundError(f"Model file '{model_name}' not found in '{models_dir}' or specified path.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化默认模型
model_name = "yolov8m-face.pt"
model = load_model(model_name).to(device)

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("800x600")
        self.model_name = model_name  # 保存模型名称
        self.image_path = None

        # 菜单栏
        menu_bar = Menu(root)
        root.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="上传图片", command=self.upload_photo)

        # 左侧框架
        self.left_frame = tk.Frame(root, width=200)
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="ns")

        self.model_select_btn = tk.Button(self.left_frame, text="模型选择", command=self.select_model)
        self.model_select_btn.pack(pady=5)

        self.execute_btn = tk.Button(self.left_frame, text="执行", command=self.execute_detection)
        self.execute_btn.pack(pady=5)

        self.save_btn = tk.Button(self.left_frame, text="保存", command=self.save_image)
        self.save_btn.pack(pady=5)

        # 右侧框架
        self.right_frame = tk.Frame(root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.horizontal_frame = tk.Frame(self.right_frame)
        self.horizontal_frame.grid(row=0, column=0, sticky="ew")

        self.image_label = tk.Label(self.horizontal_frame, text="No image")
        self.image_label.pack(padx=5)

        self.result_image = tk.Label(self.horizontal_frame, text="No result image")
        self.result_image.pack(padx=5)

        self.result_box = tk.Text(self.right_frame, height=20, width=60)
        self.result_box.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=1)

    def upload_photo(self):
        self.image_path = filedialog.askopenfilename()
        print(f"Selected image path: {self.image_path}")
        if self.image_path:
            img = Image.open(self.image_path)
            img = img.resize((250, 250))
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img, text="")
            self.image_label.image = img

    def select_model(self):
        # 弹出窗口选择模型
        self.model_window = Toplevel(self.root)
        self.model_window.title("选择模型")
        self.model_window.geometry("400x400")

        model_listbox = Listbox(self.model_window)
        model_listbox.pack(pady=10, padx=10)
        
        # 示例模型名称，可以替换为实际的模型名称列表
        models = ["YOLOv8m-face.pt", "YOLOv8n-face.pt", "yolov8l-face-lindevs.pt", "yolov8m-face-lindevs.pt", "yolov8n-face-lindevs.pt", "yolov8s-face-lindevs.pt", "yolov8x-face-lindevs.pt", "yolov8s-pose.pt"]
        for model in models:
            model_listbox.insert(tk.END, model)

        def confirm_model():
            selected_model = model_listbox.get(tk.ACTIVE)
            if selected_model:
                self.model_name = selected_model
                global model
                model = load_model(self.model_name).to(device)
            self.model_window.destroy()

        def cancel_selection():
            self.model_window.destroy()

        confirm_btn = Button(self.model_window, text="确定", command=confirm_model)
        confirm_btn.pack(side=tk.LEFT, padx=10, pady=10)

        cancel_btn = Button(self.model_window, text="取消", command=cancel_selection)
        cancel_btn.pack(side=tk.RIGHT, padx=10, pady=10)

    def execute_detection(self):
        try:
            if not self.image_path:
                messagebox.showerror("Error", "No image uploaded!")
                return

            results = model(self.image_path)

            if isinstance(results, list) and len(results) > 0:
                results = results[0]
                if hasattr(results, 'boxes') and results.boxes is not None:
                    faces = results.boxes.xyxy.cpu().numpy()                      #gpu张量需要放到cpu上转成numpy
                    print(f"Detected faces: {faces}")
                    self.result_box.delete("1.0", tk.END)                         #显示标注信息
                    count = 0
                    for face in faces:
                        count += 1
                        if self.model_name != "yolov8s-pose.pt":
                            self.result_box.insert(tk.END, f"Face{count}: {face}\n")
                        else:
                            self.result_box.insert(tk.END, f"Person{count}: {face}\n")

                    result_image = results.plot()
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB
                    result_image = Image.fromarray(result_image)                  #转化为PIL格式
                    result_image = result_image.resize((250, 250))
                    result_image = ImageTk.PhotoImage(result_image)
                    self.result_image.config(image=result_image, text="")
                    self.result_image.image = result_image  

                else:
                    print("No faces detected.")
            else:
                print("Detection failed or no faces detected.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"An error occurred: {e}")

    def save_image(self):
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

                    self.result_box.delete("1.0", tk.END)                       #显示标注信息
                    count = 0

                    for face in faces:
                        count += 1
                        self.result_box.insert(tk.END, f"Face{count}: {face}\n")

                    image = cv2.imread(self.image_path)
                    image_height, image_width = image.shape[:2]
                    count = 0

                    # 弹出对话框选择保存文件夹
                    save_dir = filedialog.askdirectory(title="Select Directory to Save Faces")
                    if not save_dir:
                        messagebox.showerror("Error", "No directory selected!")
                        return

                    for face in faces:
                        x1, y1, x2, y2 = map(int, face[:4])

                        if self.model_name != "yolov8s-pose.pt":
                            # 扩大框到原来的两倍
                            width = x2 - x1
                            height = y2 - y1
                            new_x1 = max(0, x1 - width // 2)
                            new_y1 = max(0, y1 - height // 2)
                            new_x2 = min(image_width, x2 + width // 2)
                            new_y2 = min(image_height, y2 + height // 2)

                            face_image = image[new_y1:new_y2, new_x1:new_x2]
                            face_save_path = os.path.join(save_dir, f"face_{count}.jpg")
                        else:
                            face_image = image[y1:y2, x1:x2]
                            face_save_path = os.path.join(save_dir, f"person_{count}.jpg")

                        
                        cv2.imwrite(face_save_path, face_image)
                        count += 1

                    messagebox.showinfo("Success", f"Detected faces saved as individual images in {save_dir}")
                else:
                    print("No faces detected.")
            else:
                print("Detection failed or no faces detected.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
