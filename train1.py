from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from dataloader import build_model, DataLoader
import matplotlib.pyplot as plt
import os
# 设置GPU，确保模型可以使用GPU加速
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


# 定义训练模型的函数
def train_model(model, train_generator, test_generator, epochs, save_path):

    callbacks = [
        ModelCheckpoint(filepath=save_path, monitor='val_acc', verbose=1,  mode='auto', save_best_only=True, save_weights_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    ]

    history = model.fit(
        train_generator,
        # steps_per_epoch=math.ceil(train_size / batch_size),
        # validation_steps=math.ceil(val_size / batch_size),
        steps_per_epoch=400,
        validation_steps=20,
        validation_data=test_generator,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    plot_training(history)


# 绘制训练结果图表的函数
def plot_training(history):

    plt.figure(1)
    plt.plot(history.history['acc'], marker='o')
    plt.plot(history.history['val_acc'], marker='o')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('D:/大学课件/毕业设计/Outputimage/MobileNetacc.png')
    plt.close()

    plt.figure(2)
    plt.plot(history.history['loss'], marker='o')
    plt.plot(history.history['val_loss'], marker='o')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('D:/大学课件/毕业设计/Outputimage/MobileNetloss.png')
    plt.close()

if __name__ == "__main__":
    #对参数进行定义
    datadir = 'D:/大学课件/毕业设计/Train'
    epochs = 50
    batch_size = 4
    image_hw = (224, 224)
    class_name = ['斑点落叶病', '灰斑病', '花叶病', '褐斑病', '锈病']
    model_path = 'C:/Users/Funeko/.keras/models/mobilenet_1_0_224_tf_no_top.h5'
    save_path = 'D:/大学课件/毕业设计/Model/MobileNet.h5'

    # 初始化数据加载器和模型
    data_loader = DataLoader(datadir, class_name, batch_size, image_hw)
    model = build_model(image_hw + (3,), len(class_name), model_path)
    model.compile(optimizer=Adam(1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss=categorical_crossentropy,
                  metrics=['acc'])

    # 创建数据生成器
    train_generator = DataLoader.data_generator('train', datadir, batch_size, image_hw)
    test_generator = DataLoader.data_generator('test', datadir, batch_size, image_hw)
    train_size = train_generator.samples
    val_size = test_generator.samples
    # 训练模型
    train_model(model, train_generator, test_generator, epochs, save_path)
