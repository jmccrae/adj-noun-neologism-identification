import gzip
import fileinput
from random import shuffle
from collections import Counter
from nltk import word_tokenize
from math import log

positives = [w.strip() for w in open("adj-nouns.txt").readlines()]
negatives = [w.strip() for w in open("neg-adj-nouns.txt").readlines()]

test = [[p,1] for p in positives[:1000]] + [[n,0] for n in negatives[:1000]]
dev = [[p,1] for p in positives[1000:2000]] + [[n,0] for n in negatives[1000:2000]]
training = [[p,1] for p in positives[2000:]] + [[n, 0] for n in negatives[2000:]]
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

tp = 0
fp = 0
fn = 0
tn = 0

def predict(sent):
    words = word_tokenize(sent.lower())
    score_pos = 0.0
    score_neg = 0.0
    for w in words:
        score_pos += log((count_pos[w]+1) / pos_sum)
        score_neg += log((count_neg[w]+1) / neg_sum)
    return score_pos, score_neg

for example in test:
    score_pos, score_neg = predict(example[0])
    if score_pos > score_neg and example[1] == 1:
        tp += 1
    elif score_pos > score_neg and example[1] == 0:
        fp += 1
    elif score_pos < score_neg and example[1] == 1:
        fn += 1
    else:
        tn += 1

print("Accuracy : %.3f" % ((tp+tn) / (tp+fp+fn+tn)))
print("Precision: %.3f" % (tp / (tp+fp)))
print("Recall   : %.3f" % (tp / (tp+fn)))
print("F-Measure: %.3f" % (2.0 * tp / (2.0*tp + fp + fn)))
    
