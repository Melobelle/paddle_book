#!/usr/bin/env python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: xiake(kedou1993@163.com)
Date:    2017/11/29

使用paddle框架实现逻辑数字识别案例，关键步骤如下：
1.定义分类器网络结构
2.初始化
3.配置网络结构
4.定义成本函数cost
5.定义优化器optimizer
6.定义事件处理函数
7.进行训练
8.利用训练好的模型进行预测
"""

import matplotlib

matplotlib.use('Agg')
import os
from PIL import Image
import numpy as np
import paddle.v2 as paddle
from paddle.v2.plot import Ploter

with_gpu = os.getenv('WITH_GPU', '0') != '0'

step = 0

# 绘图相关标注
train_title_cost = "Train cost"
test_title_cost = "Test cost"

train_title_error = "Train error rate"
test_title_error = "Test error rate"

data = paddle.dataset.mnist.train()
count = 2
for d in data():
    print d
    # print type(d)
    count -= 1
    if count == 0:
        break


test_arr = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0]
            ]


def read_data(data):
    def reader():
        for d in data:
            yield d[:-1], int(d[-1])
    return reader


d = read_data(np.array(test_arr))
for dd in d():
    print dd


def softmax_regression(img):
    """
    定义softmax分类器：
        只通过一层简单的以softmax为激活函数的全连接层，可以得到分类的结果
    Args:
        img -- 输入的原始图像数据
    Return:
        predict -- 分类的结果
    """
    predict = paddle.layer.fc(
        input=img, size=10, act=paddle.activation.Softmax())
    return predict


def multilayer_perceptron(img):
    """
    定义多层感知机分类器：
        含有两个隐藏层（即全连接层）的多层感知器
        其中两个隐藏层的激活函数均采用ReLU，输出层的激活函数用Softmax
    Args:
        img -- 输入的原始图像数据
    Return:
        predict -- 分类的结果
    """
    # 第一个全连接层
    hidden1 = paddle.layer.fc(input=img, size=128, act=paddle.activation.Relu())
    # 第二个全连接层
    hidden2 = paddle.layer.fc(
        input=hidden1, size=64, act=paddle.activation.Relu())
    # 第三个全连接层，需要注意输出尺寸为10,，对应0-9这10个数字
    predict = paddle.layer.fc(
        input=hidden2, size=2, act=paddle.activation.Softmax())
    return predict


def convolutional_neural_network(img):
    """
    定义卷积神经网络分类器：
        输入的二维图像，经过两个卷积-池化层，使用以softmax为激活函数的全连接层作为输出层
    Args:
        img -- 输入的原始图像数据
    Return:
        predict -- 分类的结果
    """
    # 第一个卷积-池化层
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())

    # 第二个卷积-池化层
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # 全连接层
    predict = paddle.layer.fc(
        input=conv_pool_2, size=10, act=paddle.activation.Softmax())
    return predict


def netconfig():
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(11))

    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(2))

    predict = softmax_regression(images)
    # predict = multilayer_perceptron(images)
    # predict = convolutional_neural_network(images)

    # 定义成本函数，addle.layer.classification_cost()函数内部采用的是交叉熵损失函数
    cost = paddle.layer.classification_cost(input=predict, label=label)

    # 利用cost创建参数parameters
    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.01 / 128.0,
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

    config_data = [images, label, predict, cost, parameters, optimizer]

    return config_data


def plot_init():
    """
    绘图初始化函数：
        初始化绘图相关变量
    Args:
    Return:
        cost_ploter -- 用于绘制cost曲线的变量
        error_ploter -- 用于绘制error_rate曲线的变量
    """
    # 绘制cost曲线所做的初始化设置
    cost_ploter = Ploter(train_title_cost, test_title_cost)

    # 绘制error_rate曲线所做的初始化设置
    error_ploter = Ploter(train_title_error, test_title_error)

    ploter = [cost_ploter, error_ploter]

    return ploter


def load_image(file):
    """
    定义读取输入图片的函数：
        读取指定路径下的图片，将其处理成分类网络输入数据对应形式的数据，如数据维度等
    Args:
        file -- 输入图片的文件路径
    Return:
        im -- 分类网络输入数据对应形式的数据
    """
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32).flatten()
    im = im / 255.0
    return im


def infer(predict, parameters, file):
    """
    定义判断输入图片类别的函数：
        读取并处理指定路径下的图片，然后调用训练得到的模型进行类别预测
    Args:
        predict -- 输出层
        parameters -- 模型参数
        file -- 输入图片的文件路径
    Return:
    """
    # 读取并预处理要预测的图片
    test_data = []
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_data.append((load_image(cur_dir + file),))

    # 利用训练好的分类模型，对输入的图片类别进行预测
    probs = paddle.infer(
        output_layer=predict, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)
    print "Label of image/infer_3.png is: %d" % lab[0][0]


def main():
    """
    主函数：
        定义神经网络结构，训练模型并打印学习曲线、预测测试数据类别
    Args:
    Return:
    """
    # 初始化，设置是否使用gpu，trainer数量
    paddle.init(use_gpu=with_gpu, trainer_count=1)

    # 定义神经网络结构
    images, label, predict, cost, parameters, optimizer = netconfig()

    # 构造trainer,配置三个参数cost、parameters、update_equation，它们分别表示成本函数、参数和更新公式
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    costs = []

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.pass_id % 100 == 0:
                print("Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id, event.cost))
                costs.append(event.cost)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(read_data(np.array(test_arr)), buf_size=8192),
            batch_size=128),
        event_handler=event_handler,
        num_passes=10)

    # 预测输入图片的类型

if __name__ == '__main__':
    main()
