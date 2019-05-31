import wordnet
import nltk

wn = wordnet.parse_wordnet("wn31.xml")

for entry in wn.entries:
    if " " in entry.lemma.written_form:
        tokens = nltk.word_tokenize(entry.lemma.written_form)
        tags = nltk.pos_tag(tokens)
        if (all(tags[i][1] == "JJ" for i in range(len(tags)-1)) and 
                (tags[-1][1] == "NN" or tags[-1][1] == "NNS")):
            print(entry.lemma.written_form)
