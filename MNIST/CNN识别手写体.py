import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
import MNIST

# 全局变量  
batch_size = 128  # 批处理样本数量
nb_classes = 10  # 分类数目
epochs = 600  # 迭代次数
img_rows, img_cols = 28, 28  # 输入图片样本的宽高
nb_filters = 32  # 卷积核的个数
pool_size = (2, 2)  # 池化层的大小
kernel_size = (5, 5)  # 卷积核的大小
input_shape = (img_rows, img_cols,1)  # 输入图片的维度


X_train, Y_train = MNIST.get_training_data_set(600, False)  # 加载训练样本数据集，和one-hot编码后的样本标签数据集。最大60000
X_test, Y_test = MNIST.get_test_data_set(100, False)  # 加载测试特征数据集，和one-hot编码后的测试标签数据集，最大10000
X_train = np.array(X_train).astype(bool).astype(float)/255    #数据归一化
X_train=X_train[:,:,:,np.newaxis]  # 添加一个维度，代表图片通道。这样数据集共4个维度，样本个数、宽度、高度、通道数
Y_train = np.array(Y_train)
X_test = np.array(X_test).astype(bool).astype(float)/255    #数据归一化
X_test=X_test[:,:,:,np.newaxis]  # 添加一个维度，代表图片通道。这样数据集共4个维度，样本个数、宽度、高度、通道数
Y_test = np.array(Y_test)
print('样本数据集的维度：', X_train.shape,Y_train.shape)
print('测试数据集的维度：', X_test.shape,Y_test.shape)
print(MNIST.printimg(X_train[1]))
print(Y_train[1])
# 构建模型
model = Sequential()
model.add(Conv2D(6,kernel_size,input_shape=input_shape,strides=1))  # 卷积层1
model.add(AveragePooling2D(pool_size=pool_size,strides=2))  # 池化层
model.add(Conv2D(12,kernel_size,strides=1))  # 卷积层2
model.add(AveragePooling2D(pool_size=pool_size,strides=2))  # 池化层
model.add(Flatten())  # 拉成一维数据
model.add(Dense(128))  # 全连接层1
model.add(Dense(nb_classes))  # 全连接层2
model.add(Activation('sigmoid'))  # sigmoid评分

# 编译模型
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
# 训练模型
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(X_test, Y_test))
# 评估模型
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
