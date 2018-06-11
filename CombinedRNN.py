import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/data', one_hot=True)

memory_learning_rate = 0.01
classifier_learning_rate = 0.01

input_size = 784
classes = 10

training_iterations = 4
batch_size = 100

savepath = 'model/model'
summarypath = '/Users/sriabhirath/Desktop/PythonProjs/CombinedRNN/SessionSummary'

def base_model(X, weights, biases):
    with tf.name_scope('BaseModel') as scope:
        i1 = tf.nn.tanh(tf.add(tf.matmul(X, weights['i1']), biases['i1']))
        i1 = tf.nn.dropout(i1, 0.8)
        h1 = tf.nn.tanh(tf.add(tf.matmul(i1, weights['h1']), biases['h1']))
        h1 = tf.nn.dropout(h1, 0.5)
        o1 = tf.nn.tanh(tf.add(tf.matmul(h1, weights['o1']), biases['o1']))

    return o1

def comb_model(X, weights, biases):
    with tf.name_scope('CombModel') as scope:
        i1 = tf.nn.tanh(tf.add(tf.matmul(X, weights['i1']), biases['i1']))
        h1 = tf.nn.tanh(tf.add(tf.matmul(i1, weights['h1']), biases['h1']))
        o1 = tf.nn.tanh(tf.add(tf.matmul(h1, weights['o1']), biases['o1']))

    return o1

weights_base_input = {
    'i1': tf.Variable(tf.random_normal([input_size, 300])),
    'h1': tf.Variable(tf.random_normal([300, 100])),
    'o1': tf.Variable(tf.random_normal([100, input_size]))
}
biases_base_input = {
    'i1': tf.Variable(tf.random_normal([300])),
    'h1': tf.Variable(tf.random_normal([100])),
    'o1': tf.Variable(tf.random_normal([input_size]))
}
weights_base = {
    'i1': tf.Variable(tf.random_normal([input_size*2, 300])),#Meant to be input_size *2
    'h1': tf.Variable(tf.random_normal([300, 100])),
    'o1': tf.Variable(tf.random_normal([100, classes]))
}
biases_base = {
    'i1': tf.Variable(tf.random_normal([300])),
    'h1': tf.Variable(tf.random_normal([100])),
    'o1': tf.Variable(tf.random_normal([classes]))
}
weights_comb_0 = {
    'i1': tf.Variable(tf.random_normal([input_size, 300])),
    'h1': tf.Variable(tf.random_normal([300, 100])),
    'o1': tf.Variable(tf.random_normal([100, input_size]))
}
biases_comb_0 = {
    'i1': tf.Variable(tf.random_normal([300])),
    'h1': tf.Variable(tf.random_normal([100])),
    'o1': tf.Variable(tf.random_normal([input_size]))
}
weights_comb_1 = {
    'i1': tf.Variable(tf.random_normal([input_size, 300])),
    'h1': tf.Variable(tf.random_normal([300, 100])),
    'o1': tf.Variable(tf.random_normal([100, input_size]))
}
biases_comb_1 = {
    'i1': tf.Variable(tf.random_normal([300])),
    'h1': tf.Variable(tf.random_normal([100])),
    'o1': tf.Variable(tf.random_normal([input_size]))
}
weights_comb_2 = {
    'i1': tf.Variable(tf.random_normal([input_size, 300])),
    'h1': tf.Variable(tf.random_normal([300, 100])),
    'o1': tf.Variable(tf.random_normal([100, input_size]))
}
biases_comb_2 = {
    'i1': tf.Variable(tf.random_normal([300])),
    'h1': tf.Variable(tf.random_normal([100])),
    'o1': tf.Variable(tf.random_normal([input_size]))
}
weights_comb_3 = {
    'i1': tf.Variable(tf.random_normal([input_size, 300])),
    'h1': tf.Variable(tf.random_normal([300, 100])),
    'o1': tf.Variable(tf.random_normal([100, input_size]))
}
biases_comb_3 = {
    'i1': tf.Variable(tf.random_normal([300])),
    'h1': tf.Variable(tf.random_normal([100])),
    'o1': tf.Variable(tf.random_normal([input_size]))
}

memory_net_vars = [list(weights_comb_0.values()), list(weights_comb_3.values()), list(weights_comb_2.values()), list(weights_comb_1.values()),
                   list(biases_comb_0.values()), list(biases_comb_1.values()), list(biases_comb_2.values()), list(biases_comb_3.values())]
classifier_net_vars = [list(weights_base.values()), list(biases_base.values()), list(weights_base_input.values()), list(biases_base_input.values())]


x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, classes])

base_model_initial_output = base_model(x, weights_base_input, biases_base_input)
comb_input = comb_model(base_model_initial_output, weights_comb_0, biases_comb_0)
comb1 = comb_model(comb_input, weights_comb_1, biases_comb_1)
comb2 = comb_model(tf.nn.tanh(tf.multiply(comb_input, comb1)), weights_comb_2, biases_comb_2)
comb3 = comb_model(tf.nn.tanh(tf.multiply(comb_input, comb2)), weights_comb_3, biases_comb_3)

memory = tf.nn.tanh(tf.multiply(tf.nn.tanh(tf.multiply(comb1, comb2)), comb3))
combined_input = tf.reshape(tf.stack([x, memory]), shape=[-1, 1568])

model_prediction = base_model(combined_input, weights_base, biases_base)

with tf.name_scope('MemoryNetLoss') as scope:
    memory_loss = tf.reduce_mean(tf.pow(x - memory, 2))
    tf.summary.scalar('MemoryNetLoss', memory_loss)

with tf.name_scope('ClassifierNetLoss') as scope:
    classifier_model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_prediction,
                                                                                labels=y))
    tf.summary.scalar('ClassifierNetLoss', classifier_model_loss)

recall_fn = tf.train.RMSPropOptimizer(learning_rate=memory_learning_rate).minimize(memory_loss, var_list=memory_net_vars)
classifier_training_fn = tf.train.RMSPropOptimizer(learning_rate=classifier_learning_rate).minimize(classifier_model_loss, var_list=classifier_net_vars)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(model_prediction, 1))

with tf.name_scope('AccuracyMetric') as scope:
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('AccuracyMetric', accuracy)

saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()

variable_initializer = tf.global_variables_initializer()

mean_accuracy = 0

with tf.Session() as sess:
    sess.run(variable_initializer)
    summary_writer = tf.summary.FileWriter(summarypath, graph=sess.graph)
    for step in range(1, training_iterations+1):
        num_batches = int(mnist.train.num_examples/batch_size)
        for batch in range(1, num_batches+1):
            X, Y = mnist.train.next_batch(batch_size)

            mem_loss, classifier_lss, acc, _, _ = sess.run([memory_loss, classifier_model_loss, accuracy, recall_fn, classifier_training_fn], feed_dict={x: X, y: Y})
            number_of_iters = step*batch
            mean_accuracy = (mean_accuracy * number_of_iters + acc)/(number_of_iters+1)

            print("########################################################################################")
            print("Step:", step, "Batch:", batch, "MemoryNetLoss:", mem_loss, ",Classifier Loss:", classifier_lss, ",Model Accuracy:", acc, "Mean Accuracy:", mean_accuracy)

            summary_str = sess.run(merged_summary_op, feed_dict={x: X, y: Y})

            summary_writer.add_summary(summary_str, step * num_batches + batch)
    print("Optimization Complete...")
    _, test_acc = sess.run([recall_fn, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("TestAccuracy:", str(test_acc))


