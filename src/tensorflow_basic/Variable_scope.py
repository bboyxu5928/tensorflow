import tensorflow as tf

with tf.variable_scope(name_or_scope='scope_1') as scope_x:
    w1 = tf.get_variable(name='w1',shape=[1],initializer=tf.constant_initializer(1.0))
    with tf.variable_scope(name_or_scope='scope_2'):
        w2 = tf.get_variable(name='w2',shape=[1],initializer=tf.constant_initializer(2.0))
    with tf.variable_scope('scope_3'):
            with tf.variable_scope(scope_x):
                w3_1 = tf.Variable(1.0,name='w3_1')
                w3 = tf.get_variable(name='w3',shape=[1],initializer=tf.constant_initializer())
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print("{},{}".format(w1.name, w1.eval())) 
        print("{},{}".format(w2.name, w2.eval()))
        print("{},{}".format(w3.name, w3.eval()))
        print("{},{}".format(w3_1.name, w3_1.eval()))  