import tensorflow as tf
import numpy as np
import fileinput
import gzip

vocab = {"": 0}

def read_data(file_name, n):
    data = []
    finput = fileinput.input(file_name, openhook=gzip.open)
    i = 0
    for line in finput:
        line = line.decode()
        elems = line.strip().split("\t")
        if len(elems) == 2:
            data.append(elems)
            for word in elems[0].split(" "):
                if word not in vocab:
                    vocab[word] = len(vocab)
            i += 1
            if i > n:
                break
    finput.close()
    return data

training = read_data("train-shuffled.txt.gz", 10000)
dev = read_data("dev-shuffled.txt.gz", 1000)

print(len(vocab))

max_len = 20
n_inputs = 50
n_neurons = 60
n_steps = 3
n_outputs = 2
n_vocab = len(vocab)
n_embedding = 150

learning_rate = 0.001

init_embeds = tf.random_uniform([n_vocab, n_embedding], -1.0, 1.0)
embedding = tf.Variable(init_embeds)

train_inputs = tf.placeholder(tf.int32, shape=[None, n_steps], name="train_inputs")
inputs = tf.nn.embedding_lookup(embedding, train_inputs)
inputs = tf.unstack(inputs, num=n_steps, axis=1)

y = tf.placeholder(tf.int32, [None], name="y")

#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
basic_cell = tf.keras.layers.SimpleRNNCell(units=n_neurons)
outputs, states = tf.nn.static_rnn(basic_cell, inputs, dtype=tf.float32)

outputs = tf.reshape(outputs, [-1, n_neurons * n_steps])
logits = tf.layers.dense(outputs, n_outputs, activation=tf.math.sigmoid)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

n_epochs = 1000
batch_size = 200

def make_vec(sent):
    vec = np.zeros(n_steps)
    i = 0
    for w in sent.split(" "):
        if i >= n_steps:
            break
        vec[i] = vocab[w]
        i += 1
    return vec

x_dev = [make_vec(t[0]) for t in dev]
y_dev = [int(t[1]) for t in dev]
x_train = [make_vec(t[0]) for t in training]
y_train = [int(t[1]) for t in training]

print(sum(y_train))
print(len(y_train))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(training) // batch_size):
            x_batch = [make_vec(t[0]) for t in training[iteration:iteration+batch_size]]
            y_batch = [int(t[1]) for t in training[iteration:iteration+batch_size]]
            sess.run(training_op, feed_dict={train_inputs: x_batch, y: y_batch})
        acc_train = sess.run(accuracy, feed_dict={train_inputs: x_train, y: y_train})
        acc_test = sess.run(accuracy, feed_dict={train_inputs: x_dev, y: y_dev})
        print("%d: %0.3f // %0.3f" % (epoch, acc_train, acc_test))
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in path: %s" % save_path)




