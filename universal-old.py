import gzip
import fileinput
import tensorflow as tf
import tensorflow_hub as hub
from random import shuffle

positives = [w.strip() for w in open("adj-nouns.txt").readlines()]
negatives = [w.strip() for w in open("neg-adj-nouns.txt").readlines()]
shuffle(positives)
shuffle(negatives)

dev = [[p,1] for p in positives[:1000]] + [[n,0] for n in negatives[:1000]]
training = [[p,1] for p in positives[1000:]] + [[n, 0] for n in negatives[1000:]]
shuffle(training)

with tf.Graph().as_default():
    learning_rate = 0.0001

    #n_outputs = 2
    n_outputs = 1

    x = tf.placeholder(tf.float32, shape=[None, 512], name="universal_embedding")
    y = tf.placeholder(tf.int32, [None, 1], name="y")

    logits = tf.layers.dense(x, n_outputs, activation=tf.math.sigmoid)

    #xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    #loss = tf.reduce_mean(xentropy)
    loss = tf.losses.mean_squared_error(labels=y, predictions=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    #correct = tf.nn.in_top_k(logits, y, 1)
    correct = tf.math.equal(y,tf.cast(tf.math.greater(logits, 0.5),dtype=tf.int32))
    
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    n_epochs = 1000
    batch_size = 200

    x_dev = [t[0] for t in dev]
    y_dev = [[int(t[1])] for t in dev]
    x_train = [t[0] for t in training]
    y_train = [[int(t[1])] for t in training]

    #print(sum(y_train))
    #print(len(y_train))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        x_train_eval = sess.run(embed(x_train))
        x_dev_eval = sess.run(embed(x_dev))
        for epoch in range(n_epochs):
            lsum = 0
            for iteration in range(len(training) // batch_size):
                x_batch = x_train_eval[iteration:iteration+batch_size]
                y_batch = y_train[iteration:iteration+batch_size]
                o, l, _ = sess.run([logits, loss, training_op], feed_dict={x:x_batch, y: y_batch})
                #if iteration == 0:
                #    print(",".join([str(s) for s in o[1:4]]))
                lsum += l
            acc_train = sess.run(accuracy, feed_dict={x: x_train_eval, y: y_train})
            acc_test = sess.run(accuracy, feed_dict={x: x_dev_eval, y: y_dev})
            print("%d: %0.3f // %0.3f (%.03f)" % (epoch, acc_train, acc_test, lsum))
     
