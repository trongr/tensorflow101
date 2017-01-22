# Download and load MNIST
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def main():
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10])) # Init to zeroes: what about symmetry breaking?
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10]) # ground truth probability distribution
    # Average cross entropy cross over training data in minibatch:
    # tf.nn.softmax_cross_entropy_with_logits is more stable numerically:
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    learning_rate = 0.5
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    num_iters = 1000
    for i in xrange(num_iters):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # Train on a batch of 100 and update W and b above:
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Check model accuracy on minibatch:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == "__main__":
    main()
