import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Go to thatha and spend some time with him at 4:00pm

mnist = input_data.read_data_sets('tmp/data', one_hot=True)

SAVEPATH = 'model/model.ckpt'

learning_rate = 0.001
training_iterations = 5
batch_size = 100

input_dimensions = 784
output_dimensions = 10

X = tf.placeholder(tf.float32, [None, input_dimensions])
Y = tf.placeholder(tf.float32, [None, output_dimensions])

weights = {
    'inputlayer': tf.Variable(tf.random_normal([input_dimensions, 450])),
    'hiddenlayer':tf.Variable(tf.random_normal([450, 200])),
    'outputlayer': tf.Variable(tf.random_normal([200, output_dimensions]))
}

biases = {
    'inputlayer': tf.Variable(tf.random_normal([450])),
    'hiddenlayer': tf.Variable(tf.random_normal([200])),
    'outputlayer': tf.Variable(tf.random_normal([output_dimensions]))
}

main_vars = [weights['inputlayer'], weights['hiddenlayer'], weights['outputlayer'],
             biases['inputlayer'], biases['hiddenlayer'], biases['outputlayer']]

def model(x):

    i1 = tf.add(tf.matmul(x, weights['inputlayer']), biases['inputlayer'])
    i1 = tf.nn.relu(i1)

    h1 = tf.add(tf.matmul(i1, weights['hiddenlayer']), biases['hiddenlayer'])
    h1 = tf.nn.relu(h1)

    output = tf.add(tf.matmul(h1, weights['outputlayer']), biases['outputlayer'])

    return output


logits = model(X)
prediction = tf.nn.softmax(logits)
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=Y))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function, var_list=main_vars)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

SAVER = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init)
    for step in range(1, training_iterations+1):
        total_batches = int(mnist.train.num_examples/batch_size)
        for batch in range(total_batches):
            batchx, batchy = mnist.train.next_batch(batch_size)

            loss, acc, train_count  = sess.run([cost_function, accuracy, train_op], feed_dict={X: batchx, Y: batchy})

            print("Step:", step, ",Loss:", loss, ",accuracy:", acc)

    SAVER.save(sess, SAVEPATH)

    print("Test Accuracy:", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))


