from collections import Counter
from nltk import word_tokenize
from math import log
from numpy import arange

positives = [w.strip() for w in open("adj-nouns.txt").readlines()]
negatives = [w.strip() for w in open("neg-adj-nouns.txt").readlines()]

test = [[p,1] for p in positives[:1000]] + [[n,0] for n in negatives[:1000]]
dev = [[p,1] for p in positives[1000:2000]] + [[n,0] for n in negatives[1000:2000]]
training = [[p,1] for p in positives[2000:]] + [[n, 0] for n in negatives[2000:]]

alpha = 0.1

freq = Counter()

for line in open("counts.tsv").readlines():
    line = line.strip().split("\t")
    freq[line[0].strip()] = int(line[1].strip())

for b in arange(0.0,1.0,0.1):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for t in dev:
        words = t[0].split(" ")
        score = (freq[t[0]] + alpha) *log(freq[words[0]] + alpha) + log(freq[words[1]] + alpha) - log(freq[t[0]] + alpha)
        if score > b and t[1] == 1:
            tp += 1
        elif score > b and t[1] == 0:
            fp += 1
        elif score <= b and t[1] == 1:
            fn += 1
        else:
            tn += 1
    if tp > 0 or fp > 0:
        print()
        print("Beta     : %.3f" % b)
        print("Accuracy : %.3f" % ((tp+tn) / (tp+fp+fn+tn)))
        print("Precision: %.3f" % (tp / (tp+fp)))
        print("Recall   : %.3f" % (tp / (tp+fn)))
        print("F-Measure: %.3f" % (2.0 * tp / (2.0*tp + fp + fn)))

b = 0
tp = 0
fp = 0
fn = 0
tn = 0
for t in test:
    words = t[0].split(" ")
    score = log(freq[words[0]] + alpha) + log(freq[words[1]] + alpha) - log(freq[t[0]] + alpha)
    if score > b and t[1] == 1:
        tp += 1
    elif score > b and t[1] == 0:
        fp += 1
    elif score <= b and t[1] == 1:
        fn += 1
    else:
        tn += 1
if tp > 0 or fp > 0:
    print()
    print("Beta     : %.3f" % b)
    print("Accuracy : %.3f" % ((tp+tn) / (tp+fp+fn+tn)))
    print("Precision: %.3f" % (tp / (tp+fp)))
    print("Recall   : %.3f" % (tp / (tp+fn)))
    print("F-Measure: %.3f" % (2.0 * tp / (2.0*tp + fp + fn)))

 

