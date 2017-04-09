import tensorflow as tf
import numpy as np

inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
grad_Qout_inputs1 = tf.gradients(Qout, inputs1)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

inp = np.random.random([1,16])

grad_Qout_inputs1_val = sess.run([grad_Qout_inputs1], feed_dict={inputs1:inp})
print grad_Qout_inputs1_val