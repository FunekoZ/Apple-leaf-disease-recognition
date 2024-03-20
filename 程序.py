import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from dataloader import load_image, build_model
import os

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义模型参数
image_hw = (224, 224)
class_names = ['斑点落叶病', '灰斑病', '花叶病', '褐斑病', '锈病']
n_class = len(class_names)
save_path = 'D:/大学课件/毕业设计/Model/MobileNet.h5'

# 加载模型
model = build_model(image_hw + (3,), n_class, save_path)

class ImageClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("苹果叶片病虫害识别")
        self.geometry("800x600")
        self.model = model

        self.class_names = class_names
        self.image_label = None
        self.create_widgets()

    def create_widgets(self):
        # 欢迎信息
        self.label_welcome = tk.Label(self, text="欢迎使用苹果叶片病虫害识别程序", font=('Arial', 16))
        self.label_welcome.pack(pady=20)

        # 按钮
        self.predict_button = tk.Button(self, text="选择图片进行预测", command=self.predict)
        self.predict_button.pack()

        # 预测结果
        self.result_label = tk.Label(self, text="", font=('Arial', 14))
        self.result_label.pack(pady=20)

    def predict(self):
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("图片文件", "*.png;*.jpg;*.jpeg")])
        if not file_path:  # 用户取消了选择
            return

        # 进行预测
        result, confidence = self.predict_image(file_path)
        result_text = f"预测成功！预测类别: {result}" if confidence else result
        self.result_label['text'] = result_text

        # 显示图片
        self.display_image(file_path)
    '''
    def predict_image(self, image_path, threshold=0.5):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if img is None:
            return "读取图片失败！", None
        image = load_image(img, image_hw)
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(prediction, axis=1)[0]
        max_probability = np.max(prediction)
        if max_probability < threshold:
            return "无法判断类别，请检查图片是否正确！", None
        else:
            return self.class_names[predicted_class], max_probability
    '''

    def predict_image(self, image_path, threshold=0.5):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if img is None:
            return "读取图片失败！", None
        image = load_image(img, image_hw)
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(prediction, axis=1)[0]
        max_probability = np.max(prediction)
        # 临时移除置信度阈值判断，直接输出预测结果和置信度
        return f"{self.class_names[predicted_class]}, 置信度: {max_probability:.2f}", max_probability

    def display_image(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((500, 500))  # 设置图片显示的大小
        photo = ImageTk.PhotoImage(img)
        if self.image_label is None:
            self.image_label = tk.Label(self, image=photo)
            self.image_label.image = photo
            self.image_label.pack(pady=20)
        else:
            self.image_label.configure(image=photo)
            self.image_label.image = photo

if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
