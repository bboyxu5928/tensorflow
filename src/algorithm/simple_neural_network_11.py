# -- encoding:utf-8 --

# 引入包
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
# 数据加载
mnist = input_data.read_data_sets('data/', one_hot=True)

# 构建神经网络(4层、1 input, 2 hidden，1 output)
n_unit_hidden_1 = 256  # 第一层hidden中的神经元数目
n_unit_hidden_2 = 128  # 第二层的hidden中的神经元数目
n_input = 784  # 输入的一个样本（图像）是28*28像素的
n_classes = 10  # 输出的类别数目

# 定义输入的占位符
x = tf.placeholder(tf.float32, shape=[None, n_input], name='x')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')

# 构建初始化的w和b
weights = {
    "w1": tf.Variable(tf.random_normal(shape=[n_input, n_unit_hidden_1], stddev=0.1)),
    "w2": tf.Variable(tf.random_normal(shape=[n_unit_hidden_1, n_unit_hidden_2], stddev=0.1)),
    "out": tf.Variable(tf.random_normal(shape=[n_unit_hidden_2, n_classes], stddev=0.1))
}
biases = {
    "b1": tf.Variable(tf.random_normal(shape=[n_unit_hidden_1], stddev=0.1)),
    "b2": tf.Variable(tf.random_normal(shape=[n_unit_hidden_2], stddev=0.1)),
    "out": tf.Variable(tf.random_normal(shape=[n_classes], stddev=0.1))
}

def multiplayer_perceotron(_X, _weights, _biases):
    # 第一层 -> 第二层  input -> hidden1
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    # 第二层 -> 第三层 hidden1 -> hidden2
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, _weights['w2']), _biases['b2']))
    # 第三层 -> 第四层 hidden2 -> output
    return tf.matmul(layer2, _weights['out']) + _biases['out']


# 获取预测值
act = multiplayer_perceotron(x, weights, biases)

# 构建模型的损失函数
# softmax_cross_entropy_with_logits: 计算softmax中的每个样本的交叉熵，logits指定预测值，labels指定实际值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=y))

# 使用梯度下降求解
# 使用梯度下降，最小化误差
# learning_rate: 要注意，不要过大，过大可能不收敛，也不要过小，过小收敛速度比较慢
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 得到预测的类别是那一个
# tf.argmax:对矩阵按行或列计算最大值对应的下标，和numpy中的一样
# tf.equal:是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
# 正确率（True转换为1，False转换为0）
acc = tf.reduce_mean(tf.cast(pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

# 执行模型的训练
batch_size = 100  # 每次处理的图片数
display_step = 4  # 每4次迭代打印一次
# LAUNCH THE GRAPH
with tf.Session() as sess:
    # 进行数据初始化
    sess.run(init)

    # 模型保存、持久化
    saver = tf.train.Saver()
    epoch = 0
    while True:
        avg_cost = 0
        # 计算出总的批次
        total_batch = int(mnist.train.num_examples / batch_size)
        # 迭代更新
        for i in range(total_batch):
            # 获取x和y
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x: batch_xs, y: batch_ys}
            # 模型训练
            sess.run(train, feed_dict=feeds)
            # 获取损失函数值
            avg_cost += sess.run(cost, feed_dict=feeds)

        # 重新计算平均损失(相当于计算每个样本的损失值)
        avg_cost = avg_cost / total_batch

        # DISPLAY  显示误差率和训练集的正确率以此测试集的正确率
        if (epoch + 1) % display_step == 0:
            print("批次: %03d 损失函数值: %.9f" % (epoch, avg_cost))
            feeds = {x: mnist.train.images, y: mnist.train.labels}
            train_acc = sess.run(acc, feed_dict=feeds)
            print("训练集准确率: %.3f" % train_acc)
            feeds = {x: mnist.test.images, y: mnist.test.labels}
            test_acc = sess.run(acc, feed_dict=feeds)
            print("测试准确率: %.3f" % test_acc)

            if train_acc > 0.9 and test_acc > 0.9:
                saver.save(sess, './mn/model')
                break
        epoch += 1

    # 模型可视化输出
    writer = tf.summary.FileWriter('./mn/graph', tf.get_default_graph())
    writer.close()
'''
#简写版
input_size = 784
unit_size_1 = 256
unit_size_2 = 128
n_class = 10
batch_size = 100
epoch_num = 200

mnist = input_data.read_data_sets('data/',one_hot=True)

input_x = tf.placeholder(dtype=tf.float32,shape=[None,input_size],
                       name='input_x')
y = tf.placeholder(dtype=tf.float32,shape=[None,n_class],
                       name='y_pre')
#input_data 是输入 input_size 是上一层隐层神经元的数目 output_size
#是当前层神经元的数目 active_fun是激活函数 
def layer(input_data,input_size,output_size,active_fun=None):
    w = tf.get_variable(name='w',dtype=tf.float32,shape=[input_size,output_size],
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable(name='b',dtype=tf.float32,shape=[output_size],
                        initializer=tf.random_normal_initializer())
    #output=input_data*w+b
    output = tf.add(tf.matmul(input_data,w),b)
    if active_fun==None:
        out_put = output
    else:
        #sigmoid(output)
        out_put = active_fun(output)
    return out_put

def build_net():
    with tf.variable_scope('layer1'): #隐藏层 
        layer1 = layer(input_x, input_size, unit_size_1,tf.nn.sigmoid)
    with tf.variable_scope('layer2'): #隐藏层
        layer2 = layer(layer1,unit_size_1,unit_size_2,tf.nn.sigmoid)
    with tf.variable_scope('logits'):#输出层 
        logits = layer(layer2,unit_size_2,n_class)
    return logits
#logits预测值 
logits = build_net()

#反向过程
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

#测试使用的精确度
acc_1 = tf.equal(tf.argmax(logits,axis=1),tf.argmax(y,axis=1))
acc = tf.reduce_mean(tf.cast(acc_1,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoch = 0
    while epoch < epoch_num:
        sum_loss = 0
        avg_loss = 0
        #mnist.train.num_examples mnist总样本数 
        #batch_num是一个epoch训练次数，batch_size是一次训练的样本个数 
        #注意： 正常情况下应该对原始数据进行shuffle
        batch_num = int(mnist.train.num_examples / batch_size)
        for i in range(batch_num):
            #batch_xs是mnist里的数据，batch_ys是mnist里的数据对应的label
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {input_x:batch_xs,y:batch_ys}
            #bp过程 
            sess.run(train,feeds)
            #显示出loss
            sum_loss += sess.run(loss,feeds)
            #_,loss_1 = sess.run([train,loss],feeds)
            #sum_loss += loss_1
        avg_loss = sum_loss/batch_num
        
        if (epoch + 1) % 4 == 0:
            print("批次: %03d 损失函数值: %.9f" % (epoch, avg_loss))
            feeds = {input_x: mnist.train.images, y: mnist.train.labels}
            train_acc = sess.run(acc, feed_dict=feeds)
            print("训练集准确率: %.3f" % train_acc)
            feeds = {input_x: mnist.test.images, y: mnist.test.labels}
            test_acc = sess.run(acc, feed_dict=feeds)
            print("测试准确率: %.3f" % test_acc)   
        epoch += 1