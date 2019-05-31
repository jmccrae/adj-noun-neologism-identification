from nltk import word_tokenize
import fileinput
from collections import Counter
import sys

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

terms = [p.lower().split(" ") for p in positives] + [n.lower().split(" ") for n in negatives]
term_map = {}
for t in terms:
    if t[0] not in term_map:
        term_map[t[0]] = [[t[0]]]
    term_map[t[0]].append(t)
    if t[1] not in term_map:
        term_map[t[1]] = [[t[1]]]

freq = Counter()

line_no = 0
sys.stderr.write(".")
sys.stderr.flush()

def is_in(words, tc, i):
    for j in range(len(tc)):
        if i+j >= len(words) or words[i+j] != tc[j]:
            return False
    return True

for line in fileinput.input():
    i = line.index(":")
    words = line[i+2:].strip().split(" ")
    for i,w in enumerate(words):
        if w in term_map:
            for tc in term_map[w]:
                if is_in(words, tc, i):
                    freq[" ".join(tc)] += 1
    line_no += 1
    if line_no % 10 == 0:
        sys.stderr.write(".")
        sys.stderr.flush()
        
for f, c in freq.items():
    print(f,"\t",c)
