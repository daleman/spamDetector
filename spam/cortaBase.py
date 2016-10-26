#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

"""Modulo de TP Spam Detector.

*Cargar datos:
	python tp.py File

*Reducir Dimensiones:
	python tp.py Red dimension [n=10]
	Ej:
	python tp.py Red PCA 5

*Entrenar un modelo:
	python tp.py [Gsearch] metodo [base=full]
	Ej:
	python tp.py Dtree
	python tp.py Gsearch Dtree trainX_PCA5.npy

*Cross Validation:
	python tp.py CV metodo base [cv=10]
	Ej:
	python tp.py CV Dtree trainX_PCA5.npy 20

*Predecir:
	python tp.py Test metodo base
	Ej:
	python tp.py Test Dtree testX_PCA5.npy

	metodos = Dtree, Rforest, Etree, Knn, Nbayes, Svc
	dimensiones = PCA, iPCA, ICA, RFE
"""

import sys
import time, datetime
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import (cross_val_score, train_test_split)
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import (RFE,RFECV)
from sklearn.ensemble import ( RandomForestClassifier , ExtraTreesClassifier )
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score,roc_auc_score )
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
import pickle

if __name__ == '__main__':
	print 'Cargando ham'
	ham_txt = json.load(open('./dataset_dev/ham_dev.json'))
	print 'Cargando spam'
	spam_txt = json.load(open('./dataset_dev/spam_dev.json'))

	df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
	df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]
	train, test = train_test_split(df, test_size = 0.2)
	#trainX, trainy = cargando_atributos(train)
	#testX, testy = cargando_atributos(test)
	np.save('train', train)
	np.save('test', test)

