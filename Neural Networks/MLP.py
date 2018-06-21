import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#imported required libraries

mnist = input_data.read_data_sets('tmp/data', one_hot=True)
#Downloading and extracting the mnist dataset and setting the labels to be one_hot encoded

SAVEPATH = 'model/model.ckpt'
#After the training is complete, the model will be saved to this location

learning_rate = 0.001
training_iterations = 5
batch_size = 100
#Set up training hyoer parameters

input_dimensions = 784 #Every mnist image is flattened and has a dimension of 28*28 = 784
output_dimensions = 10 #There are 10 classes of numbers 0-9

X = tf.placeholder(tf.float32, [None, input_dimensions]) #Setting up graph placeholders
Y = tf.placeholder(tf.float32, [None, output_dimensions])

weights = {
    'inputlayer': tf.Variable(tf.random_normal([input_dimensions, 450])),#Input layer will recieve an input of size 784 and give an output of size 450
    'hiddenlayer1':tf.Variable(tf.random_normal([450, 200])),#Hidden layer1 will recieve an input of size 450 and gives a output of size 200
    'hiddenlayer2':tf.Variable(tf.random_normal([200, 100])),#Hidden layer1 will recieve an input of size 450 and gives a output of size 200
    'outputlayer': tf.Variable(tf.random_normal([100, output_dimensions]))# the output layer will recieve a input of size 200 and give an output of size 10
}
#Weights defined

biases = {
    'inputlayer': tf.Variable(tf.random_normal([450])),
    'hiddenlayer1': tf.Variable(tf.random_normal([200])),
    'hiddenlayer2': tf.Variable(tf.random_normal([100])),
    'outputlayer': tf.Variable(tf.random_normal([output_dimensions]))
}
#Biases defined

def model(x):

    i1 = tf.add(tf.matmul(x, weights['inputlayer']), biases['inputlayer'])# Taking input multiplying into the corresponding weights and adding the corresponding biases
    i1 = tf.nn.relu(i1)

    h1 = tf.add(tf.matmul(i1, weights['hiddenlayer1']), biases['hiddenlayer1'])
    h1 = tf.nn.relu(h1)

    h2 = tf.add(tf.matmul(h1, weights['hiddenlayer2']), biases['hiddenlayer2'])
    h2 = tf.nn.relu(h2)

    output = tf.add(tf.matmul(h2, weights['outputlayer']), biases['outputlayer'])

    return output
#Model defined


logits = model(X)#getting output from model.
prediction = tf.nn.softmax(logits)# converting model output into a vector/list of probabilities.
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=Y))#Making a function to calculating the cost/loss of the model.
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)#Making a function to optimize the weights and biases to reduce loss.
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#Measuring model accuracy

SAVER = tf.train.Saver()# Making a variable with the tf.train.Saver() class to save our session

variable_initializer = tf.global_variables_initializer()# making a operation to initialize all the variables

with tf.Session() as sess:#Starting a tensorflow session
    sess.run(variable_initializer)#Initializing variables
    for step in range(1, training_iterations+1):# looping the specified number of training_iterations
        total_batches = int(mnist.train.num_examples/batch_size)#Calculating the number of batches
        for batch in range(total_batches):# iterating through the batches
            batchx, batchy = mnist.train.next_batch(batch_size)# obtaining the batch data

            loss, acc, train_count  = sess.run([cost_function, accuracy, train_op], feed_dict={X: batchx, Y: batchy})# measuring loss, accuracy and training the model through the sess.run command

            print("Step:", step, "Batch", batch,"Loss:", loss, ",accuracy:", acc)#Printing the step, batch, loss and accuracy

    SAVER.save(sess, SAVEPATH)#saving model

    print("Test Accuracy:", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))# testing model accuracy


