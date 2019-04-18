#encoding:utf-8




#???????回头看一下笔记

#1.get_variable只能初始化一次,而且是第一次碰到这个变量的时候被初始化
#2.若要变量共享一定要设置共享标志 reuse为True(默认的是False)
#3.一旦reuse被设置为True，get_variable的变量就不能再被初始化了，只能找已经被初始化的值
#4. Variable不能被共享，reuse标志对其无用
#5. Variable任何时候都可以被初始化 

import tensorflow as tf
with tf.variable_scope(name_or_scope='scope_1'):
#     w0 = tf.Variable(1.0,name='w')
    w0 = tf.get_variable(name='w',shape=[1])
# with tf.variable_scope(name_or_scope='scope_1',reuse=False):
with tf.variable_scope(name_or_scope='scope_1',reuse=tf.AUTO_REUSE):

    w1 = tf.get_variable(name='w',shape=[1],initializer=tf.constant_initializer(1.3))
#     w1 = tf.Variable(2.0,name='w')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(w0))
    print(sess.run(w1))