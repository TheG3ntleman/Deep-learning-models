import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#importing required libraries

mnist = input_data.read_data_sets('tmp/data', one_hot=True)
#Downloading and extracting MNIST dataset

output_classes = 10 #Defining number of classes

#Setting model hyperparameters
training_iterations = 5
learning_rate = 0.001
batchsize = 100

#Adding conv2d layer function for simplicity
def conv2d(x, w, b, strides=1):
    conv = tf.nn.conv2d(x, filter=w, strides=[1, strides, strides, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

#Adding maxpooling2d layer function for simplicity
def maxpooling2D(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

#Defining convolutional model function
def convnet(x, weights, biases):

    x = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['wc1'])
    conv1 = maxpooling2D(conv1)
    conv1 = tf.nn.leaky_relu(conv1)

    conv2 = conv2d(conv1, weights['wc2'], biases['wc2'])
    conv2 = maxpooling2D(conv2)
    conv2 = tf.nn.leaky_relu(conv2)


    conv3 = conv2d(conv2, weights['wc3'], biases['wc3'])
    conv3 = maxpooling2D(conv3)
    conv3 = tf.nn.leaky_relu(conv3)


    conv4 = conv2d(conv3, weights['wc4'], biases['wc4'])
    conv4 = maxpooling2D(conv4)
    conv4 = tf.nn.leaky_relu(conv4)

    conv5 = conv2d(conv4, weights['wc5'], biases['wc5'])
    conv5 = maxpooling2D(conv5)
    conv5 = tf.nn.leaky_relu(conv5)

    print("ShapeRaw:", conv5.get_shape().as_list())
    conv5 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])

    dense1 = tf.add((tf.matmul(conv5, weights['wd1'])), biases['wd1'])
    dense1 = tf.nn.leaky_relu(dense1)

    return tf.add(tf.matmul(dense1, weights['wd2']), biases['wd2'])

#Setting up weights and biases for the model
weights = {

    'wc1': tf.Variable(tf.random_normal(shape=[3, 3, 1, 32])),#[filter size, filter_size, inputs, outputs]
    'wc2': tf.Variable(tf.random_normal(shape=[3, 3, 32, 64])),
    'wc3': tf.Variable(tf.random_normal(shape=[3, 3, 64, 128])),
    'wc4': tf.Variable(tf.random_normal(shape=[3, 3, 128, 256])),
    'wc5': tf.Variable(tf.random_normal(shape=[3, 3, 256, 512])),
    'wd1': tf.Variable(tf.random_normal(shape=[512, 512])),
    'wd2': tf.Variable(tf.random_normal(shape=[512, output_classes]))

}

biases = {

    'wc1': tf.Variable(tf.random_normal(shape=[32])),
    'wc2': tf.Variable(tf.random_normal(shape=[64])),
    'wc3': tf.Variable(tf.random_normal(shape=[128])),
    'wc4': tf.Variable(tf.random_normal(shape=[256])),
    'wc5': tf.Variable(tf.random_normal(shape=[512])),
    'wd1': tf.Variable(tf.random_normal(shape=[512])),
    'wd2': tf.Variable(tf.random_normal(shape=[output_classes]))

}


#Graph inputs
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#setting up training mechanism
pred = convnet(X, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#Training model
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, training_iterations+1):
        for batch in range(int(mnist.train.num_examples/batchsize)):
            batchx, batchy = mnist.train.next_batch(batchsize)
            sess.run(optimizer, feed_dict={X: batchx, Y: batchy})
            loss, acc = sess.run([cost, accuracy], feed_dict={X: batchx, Y:batchy})
            print("Step:", step, ",batch number:", batch, ",Loss:", loss, ",accuracy:", acc)
    saver.save(sess, 'Save/model.ckpt')
    print("TestAccuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels}))
