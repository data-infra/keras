import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from PIL import Image
import FaceData
# 全局变量  
batch_size = 128  # 批处理样本数量
nb_classes = 40  # 分类数目
epochs = 600  # 迭代次数
img_rows, img_cols = 57, 47  # 输入图片样本的宽高
nb_filters = 32  # 卷积核的个数
pool_size = (2, 2)  # 池化层的大小
kernel_size = (5, 5)  # 卷积核的大小
input_shape = (img_rows, img_cols,1)  # 输入图片的维度

[(X_train, Y_train), (X_valid, Y_valid),(X_test, Y_test)] =FaceData.load_data('olivettifaces.gif')

X_train=X_train[:,:,:,np.newaxis]  # 添加一个维度，代表图片通道。这样数据集共4个维度，样本个数、宽度、高度、通道数
X_valid=X_valid[:,:,:,np.newaxis]  # 添加一个维度，代表图片通道。这样数据集共4个维度，样本个数、宽度、高度、通道数
X_test=X_test[:,:,:,np.newaxis]  # 添加一个维度，代表图片通道。这样数据集共4个维度，样本个数、宽度、高度、通道数

print('样本数据集的维度：', X_train.shape,Y_train.shape)
print('测试数据集的维度：', X_test.shape,Y_test.shape)






# 构建模型
model = Sequential()
model.add(Conv2D(6,kernel_size,input_shape=input_shape,strides=1))  # 卷积层1
model.add(AveragePooling2D(pool_size=pool_size,strides=2))  # 池化层
model.add(Conv2D(12,kernel_size,strides=1))  # 卷积层2
model.add(AveragePooling2D(pool_size=pool_size,strides=2))  # 池化层
model.add(Flatten())  # 拉成一维数据
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



y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)   # 获取概率最大的分类，获取每行最大值所在的列
for i in range(len(y_pred)):
    oneimg = X_test[i,:,:,0]*256
    im = Image.fromarray(oneimg)
    im.show()
    print('第%d个人识别为第%d个人'%(i,y_pred[i]))
