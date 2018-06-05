import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

input_size = 784
hidden_layer_1 = 600
hidden_layer_2 = 300
hidden_layer_3 = 100
latent_space_size = 10

learningrate = 0.005
training_iterations = 2000
batch_size = 100

weights = {

    'i1': tf.Variable(tf.random_normal([input_size, hidden_layer_1])),
    'h1': tf.Variable(tf.random_normal([hidden_layer_1, hidden_layer_2])),
    'h2': tf.Variable(tf.random_normal([hidden_layer_2, hidden_layer_3])),
    'h3': tf.Variable(tf.random_normal([hidden_layer_3, latent_space_size])),
    'h4': tf.Variable(tf.random_normal([latent_space_size, hidden_layer_3])),
    'h5': tf.Variable(tf.random_normal([hidden_layer_3, hidden_layer_2])),
    'h6': tf.Variable(tf.random_normal([hidden_layer_2, hidden_layer_1])),
    'o1': tf.Variable(tf.random_normal([hidden_layer_1, input_size]))

}

biases = {

    'i1': tf.Variable(tf.random_normal([hidden_layer_1])),
    'h1': tf.Variable(tf.random_normal([hidden_layer_2])),
    'h2': tf.Variable(tf.random_normal([hidden_layer_3])),
    'h3': tf.Variable(tf.random_normal([latent_space_size])),
    'h4': tf.Variable(tf.random_normal([hidden_layer_3])),
    'h5': tf.Variable(tf.random_normal([hidden_layer_2])),
    'h6': tf.Variable(tf.random_normal([hidden_layer_1])),
    'o1': tf.Variable(tf.random_normal([input_size]))

}

def Encoder(X):

    i1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['i1']), biases['i1']))
    d1 = tf.nn.sigmoid(tf.add(tf.matmul(i1, weights['h1']), biases['h1']))
    d2 = tf.nn.sigmoid(tf.add(tf.matmul(d1, weights['h2']), biases['h2']))
    d3 = tf.nn.sigmoid(tf.add(tf.matmul(d2, weights['h3']), biases['h3']))

    return d3

def Decoder(X):

    d4 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['h4']), biases['h4']))
    d5 = tf.nn.sigmoid(tf.add(tf.matmul(d4, weights['h5']), biases['h5']))
    d6 = tf.nn.sigmoid(tf.add(tf.matmul(d5, weights['h6']), biases['h6']))
    o1 = tf.nn.sigmoid(tf.add(tf.matmul(d6, weights['o1']), biases['o1']))

    return o1

X = tf.placeholder(tf.float32, [None, input_size])

encoder_output = Encoder(X)
decoder_output = Decoder(encoder_output)

loss_fn = tf.reduce_mean(tf.pow(X - decoder_output, 2))
training_fn = tf.train.RMSPropOptimizer(learningrate).minimize(loss_fn)

variable_initializer = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(variable_initializer)

    num_batches = int(mnist.train.num_examples/batch_size)
    for mega_training_steps in range(training_iterations):
        for step in range(num_batches):
            train_x, _ = mnist.train.next_batch(batch_size)
            loss, _ = sess.run([loss_fn, training_fn], feed_dict={X:train_x})
            print("MegaStep:", mega_training_steps, "Step:", step, "Loss:", loss)
