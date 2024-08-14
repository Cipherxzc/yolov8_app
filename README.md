# yolov8_app

> Designed by **"Tachi Virtuoso"** *Mixsoul*😎, **"Sovereign of Perfect Scores"** *Z.rrr~*😝, and their **"vegetable fan"** *Cipherxzc*😭

&emsp;&emsp;在开发过程中，数据集存储于上级目录的 `datasets` 文件夹中。我们使用 *WIDER FACE* 数据集进行模型的训练
- **`./config/` 目录**：存储了配置文件，包括训练使用的 `yaml` 文件以及应用运行过程中读取的可用模型列表配置文件
- **`./data/` 目录**：存储了我们测试应用时使用的图片，以及默认存放结果的路径等
- **`./models/` 目录**：存放了我们提供的模型，其中主要有
    - `Cipherxzc/` 存放了我们自训练的模型
    - `akanametov/` 存放了由 *akanametov* 训练的模型
    - `lindevs/` 存放了由 *lindevs* 训练的模型
- **`./runs/` 目录**：存放了训练相关的信息
    - `detect/` 训练时数据存放的文件夹
    - `YOLOv8n-face/` 模型 *YOLOv8n-face* 的各项评估结果
    - `YOLOv8m-face/` 模型 *YOLOv8m-face* 的各项评估结果
- **`./interact.py`**：应用的核心代码
- **`./test_model.py`** 测试模型的代码
- **`./train.py`**：训练模型的代码
- **`./wider_face_to_coco8.py`**：标注数据集并转换格式的代码
- **`./verify.py`** 验证数据集是否完整的代码