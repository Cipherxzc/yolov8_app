import tkinter as tk
from tkinter import filedialog, messagebox, Menu, Toplevel, Listbox, Button
from PIL import Image, ImageTk
import torch
import os
import cv2
from ultralytics import YOLO
import argparse
import json

# 模型列表
#avaiable_model = {}
#model_list = []

# 利用命令行设置初始模型和置信度
parser = argparse.ArgumentParser(description='设置初始模型和置信度')
parser.add_argument('--model', type=str, help='设置模型', default='yolov8m-face.pt')
parser.add_argument('--confidence', type=float, help='设置置信度', default='0.6')
args = parser.parse_args()
print('设置初始模型为:{}  设置初始置信度为:{}'.format(args.model, args.confidence))

#model_name = args.model
#set_confidence = args.confidence


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
    
    # 检查是否在 models 目录
    model_in_models_dir = os.path.join(models_dir, model_name)
    if os.path.isfile(model_in_models_dir):
        return YOLO(model_in_models_dir)
    
    # 在 models 目录中搜索模型文件
    for root, _, files in os.walk(models_dir):
        if model_name in files:
            return YOLO(os.path.join(root, model_name))

    raise FileNotFoundError(f"Model file '{model_name}' not found in '{models_dir}' or specified path.")



# 初始化默认模型
# model = load_model(model_name).to(device)

class FaceDetectionApp:
    def __init__(self, root):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(current_dir, "models")
        self.model_name = args.model
        self.set_confidence = args.confidence
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("800x600")
        self.image_path = None
        self.model = load_model(self.model_name).to(self.device)
        self.Model_path = None
        self.avaiable_model = {}
        self.model_list = []

        # 菜单栏
        menu_bar = Menu(root)
        root.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="上传图片", command=self.upload_photo)
        file_menu.add_command(label="保存图片", command=self.save_image)

        # 左侧框架
        self.left_frame = tk.Frame(root, width=200)
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="ns")

        self.model_select_btn = tk.Button(self.left_frame, text="模型选择", command=self.select_model)
        self.model_select_btn.pack(pady=5)

        self.execute_btn = tk.Button(self.left_frame, text="执行", command=self.execute_detection)
        self.execute_btn.pack(pady=5)

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
        #global model_list
        self.model_list.clear()

        # 获取模型路径
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.current_dir, "models")

        # 读取json文件中模型信息
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_file_path, "config")
        config_json_path = config_path + "\\available_models.json"
        print(config_json_path)
        try:
            config_json_file = open(config_json_path, 'r')
            json_content = config_json_file.read()
            self.avaiable_model = json.loads(json_content)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"An error occurred: {e}")


        illegal_model_list = []

        # 判断模型有效性并对json文件进行修改
        for (key, value) in self.avaiable_model.items():
            if key.endswith(".pt"):
                key_path = os.path.abspath(value)
                if os.path.isfile(key_path):      
                    self.model_list.append(key)
                else:
                    print("model_path: {} didn't exist".format(key_path))
                    illegal_model_list.append(key)
                    continue
            else:
                print("model_name: {} is illegal".format(key))
                illegal_model_list.append(key)
                continue
        
        for i in illegal_model_list:
            self.avaiable_model.pop(i)

        update_json_file = open(config_json_path, 'w')
        json.dump(self.avaiable_model, update_json_file, indent=4)

        # 弹出窗口选择模型
        self.model_window = Toplevel(self.root)
        self.model_window.title("选择模型")
        self.model_window.geometry("400x400")
        model_listbox = Listbox(self.model_window)
        model_listbox.pack(pady=10, padx=10)

        # 列表初始化
        for model in self.model_list:
            model_listbox.insert(tk.END, model)

        def confirm_model():
            # 读取json文件信息
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            config_file_path = os.path.join(current_file_path, "config")
            json_path = config_file_path + "\\available_models.json"
            json_file = open(json_path, 'r')
            content = json_file.read()
            avaiable_model = json.loads(content)
            print(avaiable_model)

            # 加载用户所选模型
            selected_model = model_listbox.get(tk.ACTIVE)
            if selected_model:
                self.model_name = selected_model
                if avaiable_model.get(self.model_name):
                    self.Model_path = os.path.abspath(avaiable_model[self.model_name])
                else:
                    print("wrong model name: {}".format(self.model_name))
                global model
                print(self.Model_path)
                self.model = load_model(self.Model_path).to(self.device)
            self.model_window.destroy()

        def cancel_selection():
            self.model_window.destroy()

        def select_model_file():
            # 显示文件选择框，读取模型文件路径，并转化为相对路径
            selected_model_dir = filedialog.askopenfilename()
            current_Path = os.getcwd()
            relative_path = os.path.relpath(selected_model_dir, current_Path)  
            select_model_name = os.path.basename(selected_model_dir)

            # 判断文件类型，修改json文件
            if select_model_name.endswith(".pt"):
                if select_model_name in self.avaiable_model.keys():
                    print("Model already exists.")
                    messagebox.showinfo("model error", f"model:{select_model_name} already exists.")
                else:
                    self.model_list.append(select_model_name)
                    self.avaiable_model.update({select_model_name:relative_path}) 
                    current_file_path = os.path.dirname(os.path.abspath(__file__))
                    config_file_path = os.path.join(current_file_path, "config")
                    json_path = config_file_path + "\\available_models.json"
                    json_file = open(json_path, 'w')
                    json.dump(self.avaiable_model, json_file, indent=4)
            else:
                print("model_file is illegal")
                messagebox.showinfo("model error", f"model:{select_model_name} is illegal.")

            # 更新显示列表
            model_listbox.delete(0, tk.END)
            for model in self.model_list:
                model_listbox.insert(tk.END, model)


        confirm_btn = Button(self.model_window, text="确定", command=confirm_model)
        confirm_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        cancel_btn = Button(self.model_window, text="取消", command=cancel_selection)
        cancel_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        select_btn = Button(self.model_window, text = "添加模型文件", command=select_model_file)
        select_btn.pack(side=tk.LEFT, padx=10, pady=10)

    def execute_detection(self):
        try:
            if not self.image_path:
                messagebox.showerror("Error", "No image uploaded!")
                return

            results = self.model(self.image_path)

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
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  # BGR 转换为 RGB
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

            results = self.model(self.image_path)

            if isinstance(results, list) and len(results) > 0:
                results = results[0]
                if hasattr(results, 'boxes') and results.boxes is not None:
                    faces = results.boxes.xyxy.cpu().numpy()                    #gpu张量需要放到cpu上转成numpy
                    print(f"Detected faces: {faces}")
                    confidences = results.boxes.conf.cpu().numpy()              # 获取每个检测的置信�?
                    print(f"Confidences: {confidences}")  # 打印置信度�?

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

                    for face, confidence in zip(faces, confidences):
                        if confidence >= self.set_confidence:
                            count += 1
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

                    if count > 0:
                        messagebox.showinfo("Success", f"Detected faces with confidence > {self.set_confidence} saved as individual images in {save_dir}")
                    else:
                        messagebox.showinfo("No Faces Saved", f"No faces with confidence > {self.set_confidence} detected.")
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
