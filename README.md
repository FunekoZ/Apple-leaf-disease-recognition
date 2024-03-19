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
- **Anaconda5.2 官网下载地址**：[Anaconda Download](https://www.anaconda.com/download/)
- **Python3.6.9 官网下载地址**：[Python Release 3.6.9](https://www.python.org/downloads/release/python-369/)

### 2. 安装过程

#### CUDA

选择相对应的版本下载，在线安装版和离线安装版均可。先安装基础包，再安装升级补丁。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/1.png) <!-- 图 1 选择对应版本的 CUDA 并下载 -->

使用管理员权限安装，一路默认。

安装完成之后打开命令行，输入 `nvcc -V` 查看版本，表示安装成功。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/2.png) <!-- 图 2 验证是否成功安装 -->

安装成功后，我的电脑上点右键，打开属性->高级系统设置->环境变量，可以看到系统中多了 `CUDA_PATH` 和 `CUDA_PATH_V10_0` 两个环境变量。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/3.png) <!-- 图 3 环境变量 -->

我们还需要添加如下几个变量，在系统变量中添加如下几个变量：

- `CUDA_SDK_PATH = C:\ProgramData\NVIDIACorporation\CUDA Samples\v10.0`
- `CUDA_LIB_PATH = %CUDA_PATH%\lib\x64`
- `CUDA_BIN_PATH = %CUDA_PATH%\bin`
- `CUDA_SDK_BIN_PATH = %CUDA_SDK_PATH%\bin\win64`
- `CUDA_SDK_LIB_PATH = %CUDA_SDK_PATH%\common\lib\x64`

设置完成之后，我们可以打开命令行来查看。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/4.png) <!-- 图 4 查看路径 -->

#### cuDNN

选择对应的版本下载，解压 zip，将文件夹里的内容拷贝到 CUDA 的安装目录并覆盖相应的文件夹。CUDA 默认安装目录：
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0`

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/5.png) <!-- 图 5 cuDNN 下载 -->

#### Anaconda

选择相应的版本下载，下载过程有点慢，也可以选择从清华镜像下载，点击`Anaconda3-5.2.0-Windows-x86_64.exe`安装，注意安装过程中选择“加入到系统 PATH 环境变量”。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/6.png) <!-- 图 6 Anaconda 下载 -->
![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/7.png) <!-- 图 7 添加进入环境变量 -->

#### Python

搭建 TensorFlow 环境，官网下载 Python3.6.9 安装，安装的时候勾选`AddPython 3.6 to PATH`选项，这样可以直接添加用户变量，后续不用再设置环境变量。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/8.png) <!-- 图 8 python 下载 -->

此外，Anaconda 集成了许多第三方库，也可以从 Anaconda 自带的 Python3.6安装。  

管理员模式打开命令行，输入：`conda create –n tensorflow python=3.6`  

下载 python 并安装。安装完成后，输入：`conda list`  

可以查询本地安装了哪些库，如果有需要的库没有安装上，可以运行（***为需要安装的包名称）：`conda install ***`  

如果需要更新某个包，运行：`conda update ***`  

要激活虚拟环境，输入命令（***为你的虚拟环境名）：`Activate ***`  

然后查看是否成功可以输入：`conda info –envs`

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/9.png) <!-- 图 9 激活 TensorFlow 环境 -->

#### TensorFlow

管理员模式打开命令行，输入命令 `pip install tensorflow-gpu`。  
如果选择安装 cpu 版本，命令：`pip install tensorflow`。

安装完成后命令行输入：`python`，进入 python 编辑环境，然后输入命令：`import tensorflow as tf`，如果没有报错，则表示安装成功。

注意，一定不要 tensorflow-gpu 和 tensorflow(cpu 版)一起装，如果先安装 tensorflow-gpu 再安装 tensorflow，gpu 版本直接失效。即使输入
withtf.device("/gpu":0)依旧无法使用 gpu，系统只会用 cpu 版本计算。

若已经出现此问题，不能只卸载 tensorflow，会报缺少文件的错。必须把tensorflow-gpu版的也同时卸载，然后重新只安装 tensorflow-gpu。

需要卸载 tensorflow，cmd命令行输入：`pip uninstall tensorflow`

卸载 tensorflow-gpu，输入：`pip uninstall tensorflow-gpu`

安装完成后命令行输入：python，进入 python 编辑环境，然后输入命令：`import tensorflow as tf`

与图 10 呈现结果相同就是成功安装。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/10.png) <!-- 图 10 安装成功 -->

## 三、算法说明

### 1. 训练模型代码说明

代码结构分为三大模块，分别为 `dataloader.py`，`train.py`，`test.py`。

- **dataloader.py**：用于图片的预处理和模型的构建工作。
- **train.py**：用于训练模型，最终得到训练权重文件。
- **test.py**：用于后续训练好的模型的测试工作，测试集中的图像，并将结果以混淆矩阵形式呈现。

## 四、演示

打开 Pycharm 软件，运行 `train.py` 代码，可以根据控制台输出及时调整超参数。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/11.png) <!-- 图 11 运行环境 -->

运行过程中会显示详细损失率以及正确率，如图 12 所示。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/12.png) <!-- 图 12 训练过程 -->

训练完成后即可进行测试，测试输出的混淆矩阵，如图 13 所示。

![](https://github.com/FunekoZ/Apple-leaf-disease-recognition/blob/main/Image-foder/13.png) <!-- 图 13 测试集结果 -->
