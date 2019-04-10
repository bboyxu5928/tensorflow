import numpy as np
import tensorflow as tf
x = np.linspace(0,6,100)
y = 3*x+1
w = tf.get_variable(name='w',dtype = tf.float32,shape=[1],
                    initializer=tf.random_normal_initializer())
b = tf.get_variable(name ='b',dtype = tf.float32,shape=[1],
                    initializer=tf.random_normal_initializer())
y_pre =tf.add(tf.multiply(w,x),b)
loss=tf.reduce_mean(tf.square(y_pre -y))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(800):
        sess.run(train)
        print(sess.run([loss,w,b]))
