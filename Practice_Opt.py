from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

logs_path = "./"

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Input layer
x  = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10],  name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolutional layer 1
W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

print('h_poo1:')
print(h_pool1.get_shape().as_list())

# Convolutional layer 2
W_conv2 = weight_variable([5, 5, 16, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

print('h_poo2:')
print(h_pool2.get_shape().as_list())

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])

W_fc1 = weight_variable([7 * 7 * 16, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')
y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name='y')

# Evaluation functions
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()

# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  train_writer = tf.summary.FileWriter(logs_path+'/train', graph=tf.get_default_graph())
  test_writer = tf.summary.FileWriter(logs_path+'/test', graph=tf.get_default_graph())

# max_steps = 1000
  max_steps = mnist.train.num_examples/50
  for step in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if (step % 100) == 0:
#      print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
      print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
    _, train_summary = sess.run([train_step, merged_summary_op], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    train_writer.add_summary(train_summary, step)
    test_summary = sess.run(merged_summary_op, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    test_writer.add_summary(test_summary, step)
#  print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
  train_writer.close()
  test_writer.close()
  print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

