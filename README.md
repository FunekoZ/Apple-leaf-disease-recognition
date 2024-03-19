# 算法设计说明书

## 一、环境配置

- **笔记本电脑型号**：联想拯救者 R9000K
- **显卡版本**：NVDIA RTX 3070 LAPTOP （有条件可选择 NVDIA TITAN 系列的台式机或服务器，作为深度学习平台。）
- **CUDA 版本**：代码框架版本为 CUDA10.0
- **cuDNN 版本**：7.4.x，适应 CUDA10.0 即可
- **TensorFlow 版本**：1.13.1
- **Python 版本**：3.6.9
- **Keras 版本**：2.2.4
- **备注**：若显卡为 AMD 无 GPU 环境无法成功使用本软件。必须是 NVDIA 显卡。

## 二、相关软件安装教程

### 1. 文件下载

- **CUDA9.0 官网下载地址**：[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- **CUDNN10.0 官网下载地址（需要注册）**：[cuDNN Download](https://developer.nvidia.com/rdp/cudnn-download)
- **Anacoda5.2 官网下载地址**：[Anaconda Download](https://www.anaconda.com/download/)
- **Python3.6.9 官网下载地址**：[Python Release 3.6.9](https://www.python.org/downloads/release/python-369/)

### 2. 安装过程

#### CUDA

选择相对应的版本下载，在线安装版和离线安装版均可。先安装基础包，再安装升级补丁。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/1.png) <!-- 图 1 选择对应版本的 CUDA 并下载 -->

使用管理员权限安装，一路默认。

安装完成之后打开命令行，输入 `nvcc -V` 查看版本，表示安装成功。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/2.png) <!-- 图 2 验证是否成功安装 -->

#### cuDNN

选择对应的版本下载，解压 zip，将文件夹里的内容拷贝到 CUDA 的安装目录并覆盖相应的文件夹。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/3.png) <!-- 图 5 CUDNN 下载 -->

#### Anaconda

选择相应的版本下载，安装过程中选择“加入到系统 PATH 环境变量”。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/4.png) <!-- 图 6 Anaconda 下载 -->
![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/5.png) <!-- 图 7 添加进入环境变量 -->

#### Python

官网下载 Python3.6.9 安装，安装的时候勾选“Add Python 3.6 to PATH”选项。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/6.png) <!-- 图 8 python 下载 -->

### TensorFlow

管理员模式打开命令行，输入命令 `pip install tensorflow-gpu`。如果选择安装 cpu 版本，命令：`pip install tensorflow`。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/7.png) <!-- 图 10 安装成功 -->

## 三、算法说明

### 1. 训练模型代码说明

- **dataloader.py**：用于图片的预处理和模型的构建工作。
- **train.py**：用于训练模型，最终得到训练权重文件。
- **test.py**：用于后续训练好的模型的测试工作，测试集中的图像，并将结果以混淆矩阵形式呈现。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/8.png) <!-- 图 11 运行环境 -->
![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/9.png) <!-- 图 12 训练过程 -->
![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/10.png) <!-- 图 13 测试集结果 -->

## 四、演示

- 打开 Pycharm 软件，运行 train.py 代码，可以根据控制台输出及时调整超参数。

运行过程中会显示详细损失率以及正确率。

训练完成后即可进行测试，测试输出的混淆矩阵。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/11.png) <!-- 示例图：Pycharm软件界面 -->
![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/12.png) <!-- 示例图：训练详细损失率和正确率 -->
![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/13.png) <!-- 示例图：测试输出的混淆矩阵 -->

