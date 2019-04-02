import tensorflow as tf

v1 = tf.Variable(tf.constant(3.0),name='v1')
v2 = tf.Variable(tf.constant(4.0),name='v2')
result = v1+v2

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(result)
    saver.save(sess,'./model/model.ckpt')

# 模型的提取
v1 = tf.Variable(tf.constant(1.0), name='v1')
v2 = tf.Variable(tf.constant(4.0), name='v2')
result = v1 + v2
#
saver = tf.train.Saver()
with tf.Session() as sess:
    # 会从对应的文件夹中加载变量、图等相关信息
    saver.restore(sess, './model/model.ckpt')
    print(sess.run([result]))
