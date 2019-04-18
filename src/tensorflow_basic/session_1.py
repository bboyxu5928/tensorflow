import tensorflow as tf
input_x = tf.placeholder(tf.float32, name='input_x')
input_x1 = tf.placeholder_with_default(5.0, shape=None, name='input_x1')
w = tf.Variable(1.0,name='w')
y = tf.add(input_x,w)
y1 = tf.add(input_x1,w)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(5):
        print(sess.run(y,{input_x:i}))
#         print(sess.run(y,feed_dict={input_x:i}))
        print('==')
        print(sess.run(y1))