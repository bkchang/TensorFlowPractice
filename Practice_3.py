'''
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
training_epochs = 15
batch_size = 100
display_step = 1
logs_path = './'
n_hidden_1 = 1024  	# Number of nodes
n_hidden_2 = 1024

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

def multilayer_perceptron(x, weights, biases):
    hidden_layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)
    hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['h2']), biases['b2'])
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)
    out_layer = tf.matmul(hidden_layer_2, weights['out']) + biases['out']
    return out_layer
# Set model weights
weights = {
    'h1': tf.Variable(tf.random_normal([784, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, 10]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([10]))
}

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    pred = multilayer_perceptron(x, weights, biases)
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.name_scope('SGD'):
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.007
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    train_writer = tf.summary.FileWriter(logs_path+'/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(logs_path+'/test', graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, train_summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            train_writer.add_summary(train_summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
            
            # Write logs for testing
            c_test, test_summary = sess.run([cost, merged_summary_op],
                                     feed_dict={x: mnist.test.images, y: mnist.test.labels}) 
            test_writer.add_summary(test_summary, epoch * total_batch + i)
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    train_writer.close()
    test_writer.close()

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))
