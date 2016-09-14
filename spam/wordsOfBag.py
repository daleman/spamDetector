
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfTransformer


# Leo los mails (poner los paths correctos).
print 'Cargando ham'
ham_txt = json.load(open('/media/libre/dataset_dev/ham_dev.json'))
print 'Cargando spam'
spam_txt = json.load(open('/media/libre/dataset_dev/spam_dev.json'))

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
print 'Contar frecuencias de palabras'
X_counts = count_vect.fit_transform(ham_txt+spam_txt)

print 'Ponderar frecuencias'
tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
X = tf_transformer.transform(X_counts)

clf = DecisionTreeClassifier()


print 'cross validation'
y = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]
res = cross_val_score(clf, X, y, cv=10,n_jobs = -1)
print np.mean(res), np.std(res)
