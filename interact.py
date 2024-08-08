from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import os

# 检查并加载YOLOv8模型
model = YOLO("models/Cipherxzc/YOLOv8n-face.pt")

device = torch.device(0)
model.to(device)

# 一个窗口的类
class FaceDetectionApp:
    def __init__(self, root):  # 初始化函数
        # 初始化窗口
        self.root = root
        self.root.title("Face Detection App")  # 设置窗口标题
        self.root.geometry("800x600")  # 设置窗口大小

        # 我将功能分为左侧按钮区和右侧显示区
        # 创建左侧框架
        self.left_frame = tk.Frame(root, width=200)
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="ns")  # rowspan窗口高为2行，padx左右边距10像素，pady上下边距10像素，sticky顶部底部对齐

        # 左侧按钮1：上传按钮
        self.upload_btn = tk.Button(self.left_frame, text="上传图片", command=self.upload_photo)
        self.upload_btn.pack(pady=5)  # 将上传按钮添加到左侧框架

        # 左侧按钮2：执行按钮
        self.execute_btn = tk.Button(self.left_frame, text="执行", command=self.execute_detection)
        self.execute_btn.pack(pady=5)  # 将执行按钮添加到左侧框架

        # 创建右侧框架
        self.right_frame = tk.Frame(root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # 创建横向布局的框架
        self.horizontal_frame = tk.Frame(self.right_frame)
        self.horizontal_frame.grid(row=0, column=0, sticky="ew")

        # 显示上传图片的标签
        self.image_label = tk.Label(self.horizontal_frame, text="No image")
        self.image_label.pack(side="left", padx=5)

        # 结果文本框
        self.result_box = tk.Text(self.right_frame, height=20, width=60)
        self.result_box.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # 确保右侧框架横向扩展
        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=1)

        self.image_path = None  # 存储图像路径的变量

    def upload_photo(self):
        # 打开文件对话框以选择图片
        self.image_path = filedialog.askopenfilename()
        print(f"Selected image path: {self.image_path}")  # 打印路径
        if self.image_path:
            # 打开并调整图像大小
            img = Image.open(self.image_path)
            img = img.resize((400, 400))
            img = ImageTk.PhotoImage(img)
            # 显示图像
            self.image_label.config(image=img, text="")
            self.image_label.image = img

    def execute_detection(self):
        try:
            # 检查是否已经上传了图片
            if not self.image_path:
                messagebox.showerror("Error", "No image uploaded!")
                return

            # 使用YOLOv8模型对上传的图片进行人脸检测
            results = model(self.image_path)

            # 获取检测到的对象信息
            faces = results[0].boxes.data.cpu().numpy()

            # 检查是否检测到任何人脸
            if len(faces) == 0:
                messagebox.showinfo("Result", "No face detected.")
                return

            # 清空结果文本框
            self.result_box.delete("1.0", tk.END)

            # 遍历检测到的人脸信息并将其插入到结果文本框中
            for idx, face in enumerate(faces):
                self.result_box.insert(tk.END, f"Face {idx + 1}: {face}\n")

            # 弹出对话框选择保存路径
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Result Image"
            )

            if save_path:
                # 确保路径是绝对路径
                save_path = os.path.abspath(save_path)

                # 保存检测结果到默认目录
                results[0].save(save_path)
                messagebox.showinfo("Success", f"Result image saved to: {save_path}")

        except Exception as e:
            # 捕获任何异常，并弹出错误消息框显示错误信息
            messagebox.showerror("Error", str(e))
            # 打印详细的错误信息到控制台
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    app = FaceDetectionApp(root)  # 创建应用实例
    root.mainloop()  # 运行主循环