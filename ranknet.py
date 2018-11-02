# coding=utf-8
###################### load packages #########################
import tensorflow as tf
import numpy as np
import random

###################### 构造训练数据 #########################
def get_train_data(batch_size=32):
    # 生成的数据特征维数为10,lable为前5个维度的特征 * 2 + 后五个维度的特征 * 3得到
    X1, X2 = [], []
    Y1, Y2 = [], []

    for i in range(0, batch_size):
        x1 = []
        x2 = []
        o1 = 0.0
        o2 = 0.0
        for j in range(0, 10):
            r1 = random.random()
            r2 = random.random()
            x1.append(r1)
            x2.append(r2)

            mu = 2.0
            if j >= 5: mu = 3.0
            o1 += r1 * mu
            o2 += r2 * mu
        X1.append(x1)
        Y1.append([o1])
        X2.append(x2)
        Y2.append([o2])

    return ((np.array(X1), np.array(X2)), (np.array(Y1), np.array(Y2)))

###################### 网络结构 #########################
feature_num = 10
h1_num = 10

with tf.name_scope("input"):
    x1 = tf.placeholder(tf.float32, [None, feature_num], name="x1")
    x2 = tf.placeholder(tf.float32, [None, feature_num], name="x2")

    o1 = tf.placeholder(tf.float32, [None, 1], name="o1")
    o2 = tf.placeholder(tf.float32, [None, 1], name="o2")

# 添加隐层节点
with tf.name_scope("layer1"):
    with tf.name_scope("w1"):
        w1 = tf.Variable(tf.random_normal([feature_num, h1_num]), name="w1")
        tf.summary.histogram("layer1/w1", w1)
    with tf.name_scope("b1"):
        b1 = tf.Variable(tf.random_normal([h1_num]), name="b1")
        tf.summary.histogram("layer1/b1", b1)

    # 此处没有添加激活函数
    with tf.name_scope("h1_o1"):
        h1_o1 = tf.matmul(x1, w1) + b1
        tf.summary.histogram("h1_o1", h1_o1)

    with tf.name_scope("h2_o1"):
        h1_o2 = tf.matmul(x2, w1) + b1
        tf.summary.histogram("h2_o1", h1_o2)

# 添加输出节点
with tf.name_scope("output"):
    with tf.name_scope("w2"):
        w2 = tf.Variable(tf.random_normal([h1_num, 1]), name="w2")
        tf.summary.histogram("output/w2", w2)

    with tf.name_scope("b2"):
        b2 = tf.Variable(tf.random_normal([1]))
        tf.summary.histogram("output/b2", b2)

    h2_o1 = tf.matmul(h1_o1, w2) + b2
    h2_o2 = tf.matmul(h1_o2, w2) + b2


###################### loss #########################
with tf.name_scope("loss"):

    ############### 相关性概率 ###############
    ########## 真实相关性概率 #########
    o12 = o1 - o2
    lable_p = 1 / (1 + tf.exp(-o12))

    ########## 预测相关性概率 #########
    h_o12 = h2_o1 - h2_o2
    pred = 1 / (1 + tf.exp(-h_o12))

    ################## 交叉熵 ################
    cross_entropy = -lable_p * tf.log(pred) - (1 - lable_p) * tf.log(1 - pred)
    reduce_sum = tf.reduce_sum(cross_entropy, 1)
    loss = tf.reduce_mean(reduce_sum)
    tf.summary.scalar("loss", loss)


###################### train #########################
with tf.name_scope("train_op"):
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


###################### main #########################
with tf.Session() as sess:

    ###################### 过程写入tensorboard #########################
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/", sess.graph)

    ###################### 初始化 #########################
    init = tf.global_variables_initializer()
    sess.run(init)

    ###################### 训练 #########################
    for epoch in range(0, 3000):

        ############# 获取数据 ###########
        X, Y = get_train_data()

        ############# 训练 ################
        sess.run(train_op, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})

        ############## 打印loss #############
        if epoch % 10 == 0:

            ######## 写入tensorboard ########
            summary_result = sess.run(summary_op, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
            writer.add_summary(summary_result, epoch)

            ######## loss ########
            l_v = sess.run(loss, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
            h_o12_v = sess.run(h_o12, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
            o12_v = sess.run(o12, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})

            ######## print loss ########
            print("------ epoch[%d] loss_v[%f] ------ " % (epoch, l_v))
            for k in range(0, len(o12_v)):
                print("k[%d] o12_v[%f] h_o12_v[%f]" % (k, o12_v[k], h_o12_v[k]))