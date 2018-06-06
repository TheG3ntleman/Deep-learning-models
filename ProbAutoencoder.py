"""
    I was thinking on new ways to make completely new data then I came up with this.
    I could have a classifier model and a generator model. The classifier model would be trained first
    and would clasiifiy the input images into classed then the generator network would take the probability
    vector (softmax on the output of the classifier) and generate the image based on it. Later we could create our
    own probability vectors and generate new images.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
#Imported required libraries

mnist = input_data.read_data_sets('tmp/data', one_hot=True)
#Downloading and extracting the MNIST dataset

input_size = 784 #Defining input size
num_classes = 10 #Defining number of classes
latent_space = num_classes #Number of classes is the same as latent_space_size in this model

session_savepath = 'model/model' #Save path for model

#Classifier hyperparameters
classifier_learningrate = 0.005
classifier_training_iterations = 5
classifier_batch_size = 100

#Generator hyperparameters
generator_learningrate = 0.01
generator_training_iterations = 10
generator_batch_size = 55

#Runtime Vars
train_classifier_ = False
train_generator_ = False
test_model = False
custom_tests = True

#Probability vectors for testing
custom_probability_vectors = [[1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.7, 0.3, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.6, 0.4, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.4, 0.6, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.3, 0.7, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.1, 0.9, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0]]

#Defining generator and classifier weights and biases
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

#Defining classifier network
def classifier(x):

    i1 = tf.add(tf.matmul(x, classifier_weights['inputlayer']), classifier_biases['inputlayer'])
    i1 = tf.nn.relu(i1)

    h1 = tf.add(tf.matmul(i1, classifier_weights['hiddenlayer']), classifier_biases['hiddenlayer'])
    h1 = tf.nn.relu(h1)

    output = tf.add(tf.matmul(h1, classifier_weights['outputlayer']), classifier_biases['outputlayer'])

    return output

#Defining generator network
def generator(X):
    input_layer = tf.nn.sigmoid(tf.add(tf.matmul(X, generator_weights['i1']), generator_biases['i1']))
    hidden_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, generator_weights['h1']), generator_biases['h1']))
    output_layer = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer1, generator_weights['o1']), generator_biases['o1']))

    return output_layer

#Splitting tf.Variables(s)  into classifier and generator categories
classifier_vars = [classifier_weights['inputlayer'], classifier_weights['hiddenlayer'], classifier_weights['outputlayer'],
                   classifier_biases['inputlayer'], classifier_biases['hiddenlayer'], classifier_biases['outputlayer']]

generator_vars = [generator_weights['i1'], generator_weights['h1'], generator_weights['o1'],
                   generator_biases['i1'], generator_biases['h1'], generator_biases['o1']]

#Defining graph inputs/placeholders
X = tf.placeholder(tf.float32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])

classifier_logits = classifier(X) #Obtaining classifier output
classifier_prediction = tf.nn.softmax(classifier_logits) #Applying softmax on the classifier output
classifier_loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classifier_logits,
                                                                       labels=Y)) #Evalutating classifier loss
classifier_training_fn = tf.train.GradientDescentOptimizer(classifier_learningrate).minimize(classifier_loss_fn, var_list=classifier_vars) #Defining classifer training function
classifier_correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(classifier_prediction, 1))
classifier_accuracy = tf.reduce_mean(tf.cast(classifier_correct_prediction, tf.float32)) #Set up accuracy metric

generator_output = generator(classifier_prediction) #Obtaining generator output
generator_loss_fn = tf.reduce_mean(tf.pow(X - generator_output, 2)) #Obtaining generator loss
generator_training_fn = tf.train.RMSPropOptimizer(generator_learningrate).minimize(generator_loss_fn, var_list=generator_vars) #Defining generator optimization function

probability_vector = tf.placeholder(tf.float32, [None, num_classes]) #adding probability vector graph input for direct generator input
generator_custom_output = generator(probability_vector) #Obtainging generator output for probability_vector placeholder

saver = tf.train.Saver() #Defining tf.train.Saver() to save tf.Session()

variable_initializer = tf.global_variables_initializer() #Defining operation to initialize variables

#Defining function to test model
def model_test(sess):
    test_image = np.reshape(mnist.test.images[1], [784]) #Taking sample image from the mnist dataset and reshaping it

    print("TestImage")
    plt.imshow(np.reshape(test_image, [28, 28]), origin="upper", cmap="gray") #Plotting sample image
    plt.show()

    reconstructed_image = np.reshape(sess.run(generator_output, feed_dict={X: np.reshape(test_image, [1, 784])}), [28, 28]) #Obtaining reconstructed image

    print("ReconstructedImage")
    plt.imshow(np.reshape(reconstructed_image, [28, 28]), origin="upper", cmap="gray") #Plotting reconstructed_image
    plt.show()

#Defining function for direct custom test through defined probability vectors
def custom_test(sess):
    for custom_probability_vector in custom_probability_vectors:

        input_array = np.reshape(np.array(custom_probability_vector), [1, 10])
        reconstructed_image = np.reshape(sess.run(generator_custom_output, feed_dict={probability_vector: input_array}),
                                         [28, 28])

        print("ReconstructedImage")
        plt.imshow(np.reshape(reconstructed_image, [28, 28]), origin="upper", cmap="gray")
        plt.show()

#Defining function to train classifier network
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

#Defining function to train generator network
def train_generator(sess):

    print("TrainingGenerator\n\n\n")

    for step in range(1, generator_training_iterations + 1):
        num_batches = int(mnist.train.num_examples / generator_batch_size)
        for batch in range(num_batches):
            trainX, trainY = mnist.train.next_batch(generator_batch_size)
            loss, _ = sess.run([generator_loss_fn, generator_training_fn], feed_dict={X: trainX,
                                                                                      Y: trainY})
            print("Step:", step, "Batch:", batch, "Loss:", loss)

#Defining function to restore previously saved session if any
def restore_session(sess):
    try:
        saver.restore(sess, session_savepath)
        print("Session restored")
    except:
        print("restoration failed")

#Defining function to save current session
def save_session(sess):
    try:
        saver.save(sess, session_savepath)
        print("Session saved")
    except:
        print("Failed to save session")

#Compiling model and running
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

    if custom_tests:
        custom_test(sess)
