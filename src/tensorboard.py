#encoding:utf-8
# 可视化
#将logs复制到系统盘

import tensorflow as tf

input_x = tf.placeholder(dtype=tf.float32,name='input_x')
w  = tf.Variable(1.0)
y = tf.add(input_x,w)
# 1.开始加载scalar
tf.summary.scalar('input_x', input_x)
tf.summary.scalar('w',w)
tf.summary.scalar('y',y)
# tf.summary.scalar('loss',loss)
with tf.Session() as sess:
#     合并所有的scalar
    summ_merg = tf.summary.merge_all()
#     定义summary_writer
    summ_witer = tf.summary.FileWriter('result',sess.graph)
#     生成到根目录下面 ./result
#    summ_witer = tf.summary.FileWriter('./result',sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(5):
        merg_out = sess.run(summ_merg,feed_dict={input_x:i})
        summ_witer.add_summary(merg_out, i)
    



