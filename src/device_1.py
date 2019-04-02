import tensorflow as tf
with tf.device('/cpu:0'):
    a = tf.constant(1.0,dtype=tf.float32)
    b=tf.constant(2.0,dtype=tf.float32)
    c=tf.add(a,b)
with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
    print(sess.run(c))