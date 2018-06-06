import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('tmp/data', one_hot=True)

input_size = 784
num_classes = 10
latent_space = num_classes

session_savepath = 'model/model'

classifier_learningrate = 0.005
classifier_training_iterations = 20
classifier_batch_size = 100

generator_learningrate = 0.01
generator_training_iterations = 10
generator_batch_size = 55

train_classifier_ = False
train_generator_ = False
test_model = True

classifier_weights = {
    'inputlayer': tf.Variable(tf.random_normal([input_size, 450])),
    'hiddenlayer':tf.Variable(tf.random_normal([450, 200])),
    'outputlayer': tf.Variable(tf.random_normal([200, latent_space]))
}

classifier_biases = {
    'inputlayer': tf.Variable(tf.random_normal([450])),
    'hiddenlayer': tf.Variable(tf.random_normal([200])),
    'outputlayer': tf.Variable(tf.random_normal([latent_space]))
}

generator_weights = {
    'i1': tf.Variable(tf.random_normal([num_classes, 100])),
    'h1': tf.Variable(tf.random_normal([100, 400])),
    'o1': tf.Variable(tf.random_normal([400, input_size]))
}

generator_biases = {
    'i1': tf.Variable(tf.random_normal([100])),
    'h1': tf.Variable(tf.random_normal([400])),
    'o1': tf.Variable(tf.random_normal([input_size]))
}

def classifier(x):

    i1 = tf.add(tf.matmul(x, classifier_weights['inputlayer']), classifier_biases['inputlayer'])
    i1 = tf.nn.relu(i1)

    h1 = tf.add(tf.matmul(i1, classifier_weights['hiddenlayer']), classifier_biases['hiddenlayer'])
    h1 = tf.nn.relu(h1)

    output = tf.add(tf.matmul(h1, classifier_weights['outputlayer']), classifier_biases['outputlayer'])

    return output

def generator(X):
    input_layer = tf.nn.sigmoid(tf.add(tf.matmul(X, generator_weights['i1']), generator_biases['i1']))
    hidden_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, generator_weights['h1']), generator_biases['h1']))
    output_layer = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer1, generator_weights['o1']), generator_biases['o1']))

    return output_layer

classifier_vars = [classifier_weights['inputlayer'], classifier_weights['hiddenlayer'], classifier_weights['outputlayer'],
                   classifier_biases['inputlayer'], classifier_biases['hiddenlayer'], classifier_biases['outputlayer']]

generator_vars = [generator_weights['i1'], generator_weights['h1'], generator_weights['o1'],
                   generator_biases['i1'], generator_biases['h1'], generator_biases['o1']]

X = tf.placeholder(tf.float32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])

classifier_logits = classifier(X)
classifier_prediction = tf.nn.softmax(classifier_logits)
classifier_loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classifier_logits,
                                                                       labels=Y))
classifier_training_fn = tf.train.GradientDescentOptimizer(classifier_learningrate).minimize(classifier_loss_fn, var_list=classifier_vars)
classifier_correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(classifier_prediction, 1))
classifier_accuracy = tf.reduce_mean(tf.cast(classifier_correct_prediction, tf.float32))

generator_output = generator(classifier_prediction)
generator_loss_fn = tf.reduce_mean(tf.pow(X - generator_output, 2))
generator_training_fn = tf.train.RMSPropOptimizer(generator_learningrate).minimize(generator_loss_fn, var_list=generator_vars)

saver = tf.train.Saver()

variable_initializer = tf.global_variables_initializer()

def model_test(sess):
    test_image = np.reshape(mnist.test.images[1], [784])

    print("TestImage")
    plt.imshow(np.reshape(test_image, [28, 28]), origin="upper", cmap="gray")
    plt.show()

    reconstructed_image = np.reshape(sess.run(generator_output, feed_dict={X: np.reshape(test_image, [1, 784])}), [28, 28])

    print("ReconstructedImage")
    plt.imshow(np.reshape(reconstructed_image, [28, 28]), origin="upper", cmap="gray")
    plt.show()

def train_classifier(sess):
    print("TrainingClassifier\n\n\n")
    for step in range(1, classifier_training_iterations + 1):
        num_batches = int(mnist.train.num_examples / classifier_batch_size)
        for batch in range(num_batches):
            trainX, trainY = mnist.train.next_batch(classifier_batch_size)
            loss, acc, _ = sess.run([classifier_loss_fn, classifier_accuracy, classifier_training_fn],
                                    feed_dict={X: trainX,
                                               Y: trainY})
            print("Step:", step, "Batch:", batch, "Loss:", loss, "accuracy:", acc)

    print("ClassifierTestAccuracy:",
          sess.run(classifier_accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

def train_generator(sess):

    print("TrainingGenerator\n\n\n")

    for step in range(1, generator_training_iterations + 1):
        num_batches = int(mnist.train.num_examples / generator_batch_size)
        for batch in range(num_batches):
            trainX, trainY = mnist.train.next_batch(generator_batch_size)
            loss, _ = sess.run([generator_loss_fn, generator_training_fn], feed_dict={X: trainX,
                                                                                      Y: trainY})
            print("Step:", step, "Batch:", batch, "Loss:", loss)

def restore_session(sess):
    try:
        saver.restore(sess, session_savepath)
        print("Session restored")
    except:
        print("restoration failed")

def save_session(sess):
    try:
        saver.save(sess, session_savepath)
        print("Session saved")
    except:
        print("Failed to save session")


with tf.Session() as sess:
    sess.run(variable_initializer)

    restore_session(sess)

    if train_classifier_:
        train_classifier(sess)
    if train_generator_:
        train_generator(sess)

    save_session(sess)

    if test_model:
        model_test(sess)
