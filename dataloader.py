from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras import Model
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import glob
import os
from PIL import Image

# 图像处理函数
def load_image(image_input, image_hw, augmentation=False):
    # 检查输入是否为文件路径（字符串类型）
    if isinstance(image_input, str):
        image = Image.open(image_input)
        image = np.array(image)
    # 或者输入已经是一个NumPy数组
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise ValueError("Unsupported image input type")

    # 以下是图像预处理代码
    if len(image.shape) < 3:
        image = np.tile(np.expand_dims(image, -1), [1, 1, 3])
    elif image.shape[2] > 3:
        image = image[:, :, :3]

    if image.shape[0] != image_hw[0] or image.shape[1] != image_hw[1]:
        image = cv2.resize(image, image_hw[::-1])
    return image / 255.0


# 数据加载类
class DataLoader:
    def __init__(self, datadir, class_name, batch_size=16, image_hw=(256, 256), test_size=0.2):
        self.batch_size = batch_size
        self.n_classes = len(class_name)
        self.image_hw = image_hw
        f, y = [], []
        for i in range(self.n_classes):
            fs = glob.glob(os.path.join(datadir, class_name[i], '*'))
            f.extend(fs)
            y.extend(len(fs) * [i])

        y = to_categorical(np.array(y), self.n_classes)

        N = y.shape[0]
        idx = np.random.permutation(N)
        split = int(test_size * N)
        train_idx, test_idx = idx[split:], idx[:split]
        self.train_f, self.train_y = [f[i] for i in train_idx], y[train_idx]
        self.test_f, self.test_y = [f[i] for i in test_idx], y[test_idx]

    @staticmethod
    def data_generator(data_type, datadir, batch_size, image_hw):
        datagen = ImageDataGenerator(rescale=1. / 255) if data_type == 'test' else ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        return datagen.flow_from_directory(
            datadir,
            target_size=image_hw,
            batch_size=batch_size,
            class_mode='categorical'
        )


# 构建MobileNet模型的函数
def build_model(input_shape, n_class, model_path):

    input_layer = Input(shape=input_shape)

    # 使用迁移学习模式，载入MobileNet预训练权重
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights=None)
    base_model.load_weights(model_path, by_name=True, skip_mismatch=True)

    x = base_model(input_layer)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.7)(x)
    output_layer = Dense(n_class, activation='softmax', name='predictions', kernel_regularizer=l2(0.0001))(x)

    # 构建模型
    model = Model(inputs=input_layer, outputs=output_layer)

    return model