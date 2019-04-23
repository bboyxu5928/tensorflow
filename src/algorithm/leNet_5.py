# -- encoding:utf-8 --
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from algorithm.simple_neural_network_11 import n_class

mnist =input_data.read_data_sets('data/',one_hot=True)
train_image = mnist.train.images
train_label = mnist.train.labels
test_image = mnist.test.images
test_label =mnist.test.labels
train_sample_number = mnist.train._num_examples
#每次迭代训练的样本数量
batch_size = 64
# 输入样本的维度大小
input_dim = train_image.shape[1]
# 输出的维度大小
n_classs = train_label.shape[1]
x = tf.placeholder(tf.float32, shape=[None,input_dim], name='x')
y = tf.placeholder(tf.float32, shape=[None,n_class], name='y')

def layer(input_data,style,shape,stride=None,padding='SAME',active_fun=tf.nn.relu):
    if style=='conv':
        w = tf.get_variable(name='w',dtype=tf.float32,shape=shape,
                            initializer=tf.random_normal_initializer())
        b = tf.get_variable(name='b', shape=shape, dtype=tf.float32,
                             initializer=tf.random_normal_initializer())
        layer_out = tf.nn.conv2d(input=input_data, filter=w, strides=stride, 
                                 padding=padding)
        layer_out = tf.nn.bias_add(layer_out, b)
        layer_out = active_fun(layer_out)
    if style =='pool':
        layer_out = tf.nn.max_pool(value=input_data, ksize=shape, strides=stride, 
                                   padding=padding)  
    if style == 'fc':
        w = tf.get_variable(name='w',dtype=tf.float32,shape=shape,
                            initializer=tf.random_normal_initializer())
        b = tf.get_variable(name = 'b',dtype=tf.float32,shape=shape,
                            initializer=tf.random_normal_initializer())
        layer_out = tf.add(tf.matmul(input_data,x))
        if active_fun !=None:
            layer_out = active_fun(layer_out)
    return layer_out
    
def le_net(x):
    x = tf.reshape(x,shape=[-1,28,28,1])
    with tf.variable_scope('conv1'):
        layer1 = layer(x,style='conv',shape=[3,3,1,20],stride=[1,1,1,1])
    with tf.variable_scope('pooling1'):
        layer2 = layer(layer1,style='pool',shape=[1,2,2,1],stride=[1,2,2,1])
    with tf.variable_scope('conv2'):
        layer3 = layer(layer2,style='conv',shape=[3,3,20,50],stride=[1,1,1,1])
    with tf.variable_scope('pooling2'):
        layer4 = layer(layer3,style='pool',shape=[1,2,2,1],stride=[1,2,2,1])
    with tf.variable_scope('fc1'):
        layer4 = tf.reshape(layer4,shape=[-1,7*7*50])
        layer5 = layer(layer4,style='fc',shape=[7*7*50,500])
    with tf.variable_scope('fc2'):
        logits = layer(layer5,style='fc',shape=[500,n_classs],
                       active_fun=None)
    return logits

logits = le_net(x)
# 构建损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
# train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y,axis=1))
# 正确率
acc =tf.reduce_mean(tf.cast(pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    epoch=0
    epoch_num=20
    while epoch<epoch_num:
        avg_cost=0
        total_batch = int(train_sample_number/batch_size)
#         迭代更新
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x:batch_xs,y:batch_ys}
            sess.run(train,feed_dict=feeds)
            avg_cost+=sess.run(cost,feed_dict=feeds)
        avg_cost=avg_cost/total_batch
        
        if(epoch+1)%2==0:
            print("批次：%03d 损失函数值：%。9f" %(epoch,avg_cost))
            feeds = {x:batch_xs,y:batch_ys}
            train_acc=sess.run(acc,feed_dict=feeds)
            print("训练集准确率：%.3f" % train_acc)
            feeds={x:test_image,y:test_label}
            test_acc=sess.run(acc,feed_dict=feeds)
            print("测试数据集：%.3f" % test_acc)
        epoch+=1
print('程序结束=====')