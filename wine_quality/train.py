#!usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib
import numpy as np
import paddle.v2 as paddle

import matplotlib.pyplot as plt

import os
import csv



cur_dir = os.path.dirname(os.path.realpath(__file__))

filename = cur_dir+"/data/winequality-red.csv"


def load_data(filename, ratio):
    with open(filename) as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append([i for i in row[0].split(';')])
        data = np.array(data[1:]).astype(np.float32)
        data_num = len(data)
        slice = int(ratio * data_num)
        train_set = data[:slice]
        test_set = data[slice:]
        print train_set[0]
    return train_set, test_set

train_set, test_set = load_data(filename, 0.8)

print test_set.shape

data_dim = train_set.shape[1] - 1

print data_dim

test_arr = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0]
            ]


def read_data(data):
    def reader():
        for d in data:
            yield d[:-1], int(d[-1])
    return reader


d = read_data(train_set)
count = 10
for dd in d():
    print dd
    count -= 1
    if count==0:
        break

paddle.init(trainer_count=1, use_gpu=False)

x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(data_dim))

norm1 = paddle.layer.batch_norm(input=x, act=paddle.activation.Relu())

h1 = paddle.layer.fc(input=norm1, size=10, act=paddle.activation.Relu())

norm2 = paddle.layer.batch_norm(input=h1, act=paddle.activation.Relu())

predict = paddle.layer.fc(input=norm2, size=10, act=paddle.activation.Softmax())

label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(10))

cost = paddle.layer.cross_entropy_cost(input=predict, label=label)

parameters = paddle.parameters.create(cost)

optimizer = paddle.optimizer.Adam(
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08
)

# optimizer = paddle.optimizer.AdaDelta(rho=0.95, epsilon=1e-06)

# optimizer = paddle.optimizer.Momentum(momentum=0.95, learning_rate=0.00001)


feeding = {
    'x': 0,
    'label': 1
}

lists = []


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.pass_id % 100 == 0:
            print("pass {}, batch_id {}, cost {}").format(event.pass_id, event.batch_id, event.cost)
            result = trainer.test(reader=paddle.batch(
                read_data(test_set), batch_size=128))
            print("Test with Pass {}, Cost {}").format(
                event.pass_id, result.cost, result.metrics)
            lists.append((event.pass_id, result.cost, result.metrics))

trainer = paddle.trainer.SGD(
    cost=cost, parameters=parameters, update_equation=optimizer
)

trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(read_data(train_set), buf_size=8192),
        batch_size=128
    ),
    feeding=feeding,
    event_handler=event_handler,
    num_passes=100
)

best = sorted(lists, key=lambda list: float(list[1]))[0]
print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
# print 'The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100)

data_creater = read_data(test_set)

test_x = []
test_y = []
for item in data_creater():
    test_x.append((item[0], ))
    test_y.append(item[1])

probs = paddle.infer(
        output_layer=predict, parameters=parameters, input=test_x)

lab = np.argsort(-probs)

for i in range(20):
    print("label: {}, predict: {}").format(lab[i][0], test_y[i])

