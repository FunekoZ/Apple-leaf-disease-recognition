import os
import numpy as np
import cv2
import xlwt
from dataloader import load_image, build_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义模型参数
image_hw = (224, 224)
class_names = ['斑点落叶病', '灰斑病', '花叶病', '褐斑病', '锈病']
n_class = len(class_names)
save_path = 'D:/大学课件/毕业设计/Model/MobileNet.h5'
model = build_model(image_hw + (3,), n_class, save_path)

# 测试图像的路径
test_paths = {
    '斑点落叶病': "D:/大学课件/毕业设计/mTest/b",
    '灰斑病': "D:/大学课件/毕业设计/mTest/hui",
    '花叶病': "D:/大学课件/毕业设计/mTest/hua",
    '褐斑病': "D:/大学课件/毕业设计/mTest/he",
    '锈病': "D:/大学课件/毕业设计/mTest/xiu",
}

# 初始化混淆矩阵
confusion_matrix = np.zeros((n_class, n_class), dtype=int)

# 分类并计数函数
def classify_and_count(path, class_idx, image_hw):
    files = os.listdir(path)
    for file in files:
        img = cv2.imdecode(np.fromfile(os.path.join(path, file), dtype=np.uint8), -1)
        image = load_image(img, image_hw)
        prediction = model.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(prediction, axis=1)[0]
        confusion_matrix[class_idx][predicted_class] += 1

# 对每个类别的图片进行处理
for idx, class_name in enumerate(class_names):
    classify_and_count(test_paths[class_name], idx, image_hw)

# 创建Excel工作簿和工作表
book = xlwt.Workbook()
sheet = book.add_sheet('Confusion Matrix')

# 写入类别名称作为表头和行首
for i, class_name in enumerate(class_names):
    sheet.write(0, i+1, class_name)
    sheet.write(i+1, 0, class_name)

# 填充混淆矩阵数据到表格中
for i in range(n_class):
    for j in range(n_class):
        sheet.write(i+1, j+1, int(confusion_matrix[i][j]))

# 计算并填充每个类别的精确率到表格中
for i in range(n_class):
    accuracy = confusion_matrix[i][i] / np.sum(confusion_matrix[i]) if np.sum(confusion_matrix[i]) > 0 else 0
    sheet.write(i+1, n_class+1, accuracy)

# 保存Excel文件
excel_save_path = 'D:/大学课件/毕业设计/Excel/confusion_matrix.xls'
book.save(excel_save_path)