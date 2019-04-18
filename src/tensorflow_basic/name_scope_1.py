import tensorflow as tf
with tf.name_scope('name_scope_1'):
    a = tf.Variable(3,name='a')
    b = tf.get_variable('b',shape=[3,2],initializer=tf.random_uniform_initializer())
with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    print("{},{}".format(a.name,a.eval()))
    print("{},{}".format(b.name,b.eval()))