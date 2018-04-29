import numpy
import pandas
from PIL import Image
from keras import backend as K
from keras.utils import np_utils


"""
加载图像数据的函数,dataset_path即图像olivettifaces的路径
加载olivettifaces后，划分为train_data,valid_data,test_data三个数据集
函数返回train_data,valid_data,test_data以及对应的label
"""

# 400个样本，40个人，每人10张样本图。每张样本图高57*宽47，需要2679个像素点。每个像素点做了归一化处理
def load_data(dataset_path):
    img = Image.open(dataset_path)
    img_ndarray = numpy.asarray(img, dtype='float64') / 256
    print(img_ndarray.shape)
    faces = numpy.empty((400,57,47))
    for row in range(20):
        for column in range(20):
            faces[row * 20 + column] = img_ndarray[row * 57:(row + 1) * 57, column * 47:(column + 1) * 47]
    # 设置400个样本图的标签
    label = numpy.empty(400)
    for i in range(40):
        label[i * 10:i * 10 + 10] = i
    label = label.astype(numpy.int)
    label = np_utils.to_categorical(label, 40)  # 将40分类类标号转化为one-hot编码

    # 分成训练集、验证集、测试集，大小如下
    train_data = numpy.empty((320, 57,47))   # 320个训练样本
    train_label = numpy.empty((320,40))   # 320个训练样本，每个样本40个输出概率
    valid_data = numpy.empty((40, 57,47))   # 40个验证样本
    valid_label = numpy.empty((40,40))   # 40个验证样本，每个样本40个输出概率
    test_data = numpy.empty((40, 57,47))   # 40个测试样本
    test_label = numpy.empty((40,40))  # 40个测试样本，每个样本40个输出概率

    for i in range(40):
        train_data[i * 8:i * 8 + 8] = faces[i * 10:i * 10 + 8]
        train_label[i * 8:i * 8 + 8] = label[i * 10:i * 10 + 8]
        valid_data[i] = faces[i * 10 + 8]
        valid_label[i] = label[i * 10 + 8]
        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    return [(train_data, train_label), (valid_data, valid_label),(test_data, test_label)]


if __name__ == '__main__':
    [(train_data, train_label), (valid_data, valid_label), (test_data, test_label)] = load_data('olivettifaces.gif')
    oneimg = train_data[0]*256
    print(oneimg)
    im = Image.fromarray(oneimg)
    im.show()