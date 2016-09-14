import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.cross_validation import cross_val_score

ham = json.load(open('../dataset_dev/ham_dev.json'))
spam = json.load(open('../dataset_dev/spam_dev.json'))

cws = Counter()
cwh = Counter()

ccs = Counter()
cch = Counter()

for s in spam:
	for w in s.split(" "):
		cws[w] += 1
		for l in w:
			ccs[l] += 1

for s in ham:
	for w in s.split(" "):
		cwh[w] += 1
		for l in w:
			cch[l] += 1

cw = cws.most_common()
cc = ccs.most_common()

h = 500

for t in cw.keys():
	h = h-1
	if t in cwh.keys():
		print str(t) + ": " + cw[t] + " | " + cwh[t] + " (" + cw[t] / cwh[t] + ")"
	else:
		print str(t) + ": " + cw[t]
	if h <= 0:
		break
#c2 = counter.most_common()

#for t in c2:
#	print t
