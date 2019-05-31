import fileinput
import gzip
import nltk

trues = set([line.strip() for line in open("adj-nouns.txt").readlines()])

total = 0

for line in fileinput.input("/home/jmccrae/data/wiki/enwiki-filt.gz", openhook=gzip.open):
    line = line.decode()
    words = line[line.index(":")+1:].strip().split(" ")
    words = [w for w in words if w != ""]
    tags = nltk.pos_tag(words)
    current = ""

    for tag in tags:
        if tag[1] == "JJ":
            if current != "":
                current = current + " "
            current = current + tag[0]
        elif ((tag[1] == "NN" or tag[1] == "NNS") and current != ""):
            word = current + " " + tag[0]
            if word not in trues:
                print(word)
                current = ""
                total += 1
        else:
            current = ""
    if total > 100000:
        break

