import fileinput
import re

prog = re.compile(".*\w{3}.*")

for line in fileinput.input():
    words = line.strip().split(" ")
    if len(words) == 2 and prog.match(words[0]) and prog.match(words[1]):
        print(line.strip())
