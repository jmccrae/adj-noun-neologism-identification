import tensorflow as tf
from nltk import word_tokenize
from random import shuffle
from collections import Counter
import numpy as np
from math import log
import fileinput

vocab = {"":0}

positives = [w.strip() for w in open("adj-nouns.txt").readlines()]
negatives = [w.strip() for w in open("neg-adj-nouns.txt").readlines()]
for p in positives:
    words = word_tokenize(p.lower().strip())
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)

for p in negatives:
    words = word_tokenize(p.lower().strip())
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)

test = [[p,1] for p in positives[:1000]] + [[n,0] for n in negatives[:1000]]
dev = [[p,1] for p in positives[1000:2000]] + [[n,0] for n in negatives[1000:2000]]
training = [[p,1] for p in positives[2000:]] + [[n, 0] for n in negatives[2000:len(positives)]]
shuffle(training)

count_pos = Counter()
count_neg = Counter()

for example in training:
    words = word_tokenize(example[0].lower())
    if example[1] == 1:
        for w in words:
            count_pos[w] += 1
    else:
        for w in words:
            count_neg[w] += 1

pos_sum = sum(count_pos.values())
neg_sum = sum(count_neg.values())

n_steps = 2
n_inputs = n_steps * 2
n_outputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name ="y")

n_embedding = 300

def load_glove(vocab, n_embedding):
    W = np.random.rand(len(vocab), n_embedding) * 2 - 1
    for line in fileinput.input("glove.6B.%dd.txt" % n_embedding):
        word = line.split(" ")[0]
        if word in vocab:
            W[vocab[word],] = [float(f) for f in line.strip().split(" ")[1:]]
    return W.astype('float32')


with tf.name_scope("dnn"):
    e = tf.placeholder(tf.int32, shape=[None, n_steps], name="train_inputs")
    #logits = tf.layers.dense(X, n_outputs,  name="outputs")
    W = tf.constant(load_glove(vocab, n_embedding), name="W")
    inputs = tf.nn.embedding_lookup(W, e)
    inputs = tf.reshape(inputs, [-1, n_embedding * n_steps])
    inputs = tf.concat([inputs,X], 1)

    logits = tf.layers.dense(inputs, n_outputs, name="outputs")

alpha = 100

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 200

def make_nb(sent):
    vec = np.zeros(n_steps * 2)
    i = n_steps - 1
    for w in word_tokenize(sent.lower()):
        if i < 0:
            break
        vec[2*i] = log((count_pos[w]+1)/pos_sum)
        vec[2*i + 1] = log((count_neg[w]+1)/neg_sum)
        i -= 1
    return vec
#def make_nb(sent):
#    vec = np.zeros(n_steps)
#    i = 0
#    for w in word_tokenize(sent.lower()):
#        if i >= n_steps:
#            break
#        vec[i] = log((count_pos[w] + 1)) - log((count_neg[w] + 1))
#        i += 1
#    return vec

def make_vec(sent):
    vec = np.zeros(n_steps)
    i = n_steps - 1
    for w in word_tokenize(sent.lower()):
        if i < 0:
            break
        vec[i] = vocab[w]
        i -= 1
    return vec

e_test = [make_vec(t[0]) for t in test]
y_test = [int(t[1]) for t in test]
X_test = [make_nb(t[0]) for t in test]
e_dev = [make_vec(t[0]) for t in dev]
y_dev = [int(t[1]) for t in dev]
X_dev = [make_nb(t[0]) for t in dev]
y_train = [int(t[1]) for t in training]
X_train = [make_nb(t[0]) for t in training]
e_train = [make_vec(t[0]) for t in training]

num_examples = len(training)

batch_size = 50
 
from math import exp

best_dev = 0

def eval_from_logits(logits, y_test):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(logits)):
        if logits[i][0] > logits[i][1] and y_test[i] == 0:
            tn += 1
        elif logits[i][0] > logits[i][1] and y_test[i] == 1:
            fn += 1
        elif logits[i][0] <= logits[i][1] and y_test[i] == 0:
            fp += 1
        else:
            tp += 1
    return tp, fp, fn, tn


with tf.Session() as sess:
    init.run()
    acc_dev = accuracy.eval(feed_dict={X:X_dev, y:y_dev, e:e_dev})
    print("Init Dev accuracy:", acc_dev, "Loss:", loss.eval(feed_dict={X:X_dev, y:y_dev, e:e_dev}))
    for epoch in range(n_epochs):
        for iteration in range(num_examples // batch_size):
            X_batch = X_train[(iteration * batch_size):(iteration + 1)*batch_size]
            y_batch = y_train[(iteration * batch_size):(iteration + 1)*batch_size]
            e_batch = e_train[(iteration * batch_size):(iteration + 1)*batch_size]
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch, e:e_batch})
        acc_train = accuracy.eval(feed_dict={X:X_train, y:y_train, e:e_train})
        acc_dev = accuracy.eval(feed_dict={X:X_dev, y:y_dev, e:e_dev})
        if acc_dev > best_dev:
            best_dev = acc_dev
            l = logits.eval(feed_dict={X:X_test, y:y_test, e:e_test})
            tp, fp, fn, tn = eval_from_logits(l, y_test)


        print(epoch, "Train accuracy:", acc_train, "Dev accuracy:", acc_dev, "Loss:", 
                loss.eval(feed_dict={X:X_dev, y:y_dev, e:e_dev}))
print("Accuracy : %.3f" % ((tp+tn) / (tp+fp+fn+tn)))
print("Precision: %.3f" % (tp / (tp+fp)))
print("Recall   : %.3f" % (tp / (tp+fn)))
print("F-Measure: %.3f" % (2.0 * tp / (2.0*tp + fp + fn)))
 
