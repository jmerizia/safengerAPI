# First MNIST Test

# config=tf.ConfigProto(
#         inter_op_parallelism_threads=4,
#         intra_op_parallelism_threads=4)

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist.train.images = 55000x784 array of floats (values 0 to 1)
# mnist.train.labels = 55000x10 array of floats

# set initial model variables
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# forward propagation (interesting part):
y = tf.nn.softmax(tf.matmul(x, W) + b)
# matmul = "Matrix Multiplication"
# softmax is the activation function

y_ = t.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(
    0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32))

print sess.run(
    accuracy,
    feed_dict={x: mnist.test.images, y_:mnist.test.labels})

sess.close()