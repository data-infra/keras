from keras.models import Sequential  #引入引入网络模型
from keras.layers import Dense, Activation  # 引入全连接层和激活器
from keras.optimizers import SGD  # 引入优化器


# ====================网络模型搭建=========================
#Sequential是一系列网络层按顺序构成的栈
model = Sequential()
#将一些网络层通过.add()堆叠起来，就构成了一个模型
# 常用层（Core）、卷积层（Convolutional）、池化层（Pooling）、局部连接层、递归层（Recurrent）、嵌入层（ Embedding）、高级激活层、规范层、噪声层、包装层，当然也可以编写自己的层
model.add(Dense(units=64, activation='relu', input_dim=100))  # 全连接层100个节点,activation该层的激活器
# model.add(Activation("relu"))   # 激活函数relu
model.add(Dense(units=10,trainable=False))   # 全连接层10个节点，trainable=False表示此层的权重不进行更新
model.add(Activation("softmax"))  # 激活函数softmax
model.pop()  #删除最后一层模型



# =====================训练模型搭建=========================
# 完成模型的搭建后，我们需要使用.compile()方法来编译模型：
# loss损失函数，交叉熵。optimizer优化器，sgd随机梯度下降法进行网络训练，metrics评估模型，accuracy准确率作为评判结果
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 优化器
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)   # lr学习速率，momentum表示动量项，decay是学习速率的衰减系数(每个epoch衰减一次)，Nesterov的值是False或者True，表示使不使用Nesterov momentum
# 目标函数（损失函数）：mean_squared_error，mean_absolute_error，squared_hinge，hinge，binary_crossentropy对数损失函数，categorical_crossentropy多分类的对数损失函数
model.compile(loss='categorical_crossentropy', optimizer=sgd)


# ========================模型训练=============================
# 完成模型编译后，我们在训练数据上按batch进行一定次数的迭代来训练网络
model.fit(x_train, y_train, epochs=5, batch_size=32,shuffl=True)  #epochs迭代次数，batch_size每次迭代使用的样本数，shuffl训练集是否洗乱
# model.train_on_batch(x_batch, y_batch)   # 一次训练一个样本，可以用来处理超过机器内存的数据集

# 设置当损失函数不再下降时就停止训练
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)  # 提前结束训练的触发函数
hist = model.fit(x_train, y_train, validation_split=0.2, callbacks=[early_stopping])  # validation_split交叉验证的分割比， callbacks每次训练后的回调函数
print(hist.history) # 打印训练过程中损失函数的值以及其他度量指标

# ============================模型评估============================
# 模型评估
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)


# ==============================预测===============================
#预测新对象
classes = model.predict(x_test, batch_size=128)

# 打印获取预测时，中间层的输出
from keras import backend as K
# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[3].output])
layer_output = get_3rd_layer_output([x_test])[0]


# =========================模型存储和加载=======================

# 保存模型成文件HDF5文件。该文件将包含：1、模型的结构，以便重构该模型，2、模型的权重，3、训练配置（损失函数，优化器等）4、优化器的状态，以便于从上次训练中断的地方开始
model.save('DNN.h5')

# 从文件中加载模型
from keras.models import load_model
model = load_model('my_model.h5')

# 只保存模型的结构，而不包含其权重或配置信息
# 保存成json
json_string = model.to_json()
from keras.models import model_from_json
model = model_from_json(json_string)# 从文件中加载模型


# 保存成YAML
yaml_string = model.to_yaml()
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)  # 从文件中加载模型

# 如果需要保存模型的权重，可通过下面的代码利用HDF5进行保存
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')