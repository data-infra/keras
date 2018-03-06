from keras.models import Sequential   #引入引入网络模型
from keras.layers import Dense, Dropout  # 引入全连接层，输出层，激活器
from keras.optimizers import SGD  # 随机梯度下降法
import keras

# 产生训练和测试数据
import numpy as np
x_train = np.random.random((1000, 20))   # 产生随机数，1000个样本，20个属性
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)  # one-hot编码，1000个样本，10种分类

x_test = np.random.random((100, 20))    # 产生随机数，100个样本，20个属性
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)   # one-hot编码，100个样本，10种分类

model = Sequential()  # 序贯模型

model.add(Dense(64, activation='relu', input_dim=20))  # 20个输入，64个隐藏节点，relu激活的全连接网络
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))   # 64个输出节点的
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # 10个输出节点的

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(x_train, y_train,epochs=20,batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)