#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

"""Modulo de TP Spam Detector.

*Cargar datos:
	python tp.py File [dimension [n=10]]
*Entrenar un modelo:
	python tp.py [Gsearch] metodo [cv=10]
*Predecir:
	python tp.py Test metodo base

	metodos = Dtree, Rforest, Etree, Knn, Nbayes, Svc 
	dimensiones = PCA, iPCA, ICA, RFE
"""

import sys
import time
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

dnames = ['len','count_td','count_by','count_ESMTP','count_menora', 'count_n','count_upper','count_mm','count_ref','count_guionba'
,'count_menorp','count_px','count_from','count_id', 'count_spaces','count_cum','count_viagra','count_sex','count_vagina','count_penis'
,'count_money','count_earn','count_free','count_now','count_help','count_excl','count_preg','count_dol','count_dollar','count_dollars'
,'count_1','count_2','count_3','count_4','count_5','count_6','count_7','count_8','count_9','count_0'
,'count_work','count_arr','count_hash','count_and','count_apare','count_acor','count_plus','count_mult','count_porc','count_equal'
,'count_dot','count_dotc','count_apos','count_com','count_send','count_menor','count_dosp','count_offer','count_deal','count_join'
,'count_vote','count_weig','count_lose','count_menos','count_sombrero','count_aparen','count_cparen','count_num','count_title','count_helvetica'
,'count_arial','count_nigeria','count_win','count_FREE','count_VIAGRA','count_SEX','count_VAGINA','count_PENIS','count_MONEY','count_EARN'
,'count_NOW','count_html','count_pill','count_hours','lprom','lmax','count_solid','count_font','count_tr','count_microsoft'
,'count_HTML','count_0600','count_2002','count_nahou','count_with','count_your','count_0800','count_unv','count_sat','count_sun'
,'count_huge']

def count_mm(txt): return txt.count("mailman.enron.com")
def count_by(txt): return txt.count("by")
def count_td(txt): return txt.count('<td')
def count_font(txt): return txt.count("<font")
def count_tr(txt): return txt.count("<tr>\n")
def count_menor(txt): return txt.count("<")
def count_menora(txt): return txt.count("<a")
def count_menorp(txt): return txt.count("<p")
def count_n(txt): return txt.count("\n")
def count_spaces(txt): return txt.count(" ")
def count_viagra(txt): return txt.count("viagra")
def count_sex(txt): return txt.count("sex")
def count_vagina(txt): return txt.count("vagina")
def count_penis(txt): return txt.count("penis")
def count_money(txt): return txt.count("money")
def count_earn(txt): return txt.count("earn")
def count_free(txt): return txt.count("free")
def count_now(txt): return txt.count("now")
def count_VIAGRA(txt): return txt.count("VIAGRA")
def count_SEX(txt): return txt.count("SEX")
def count_VAGINA(txt): return txt.count("VAGINA")
def count_PENIS(txt): return txt.count("PENIS")
def count_MONEY(txt): return txt.count("MONEY")
def count_EARN(txt): return txt.count("EARN")
def count_FREE(txt): return txt.count("FREE")
def count_NOW(txt): return txt.count("NOW")
def count_help(txt): return txt.count("help")
def count_excl(txt): return txt.count("!")
def count_lose(txt): return txt.count("lose")
def count_weig(txt): return txt.count("weight")
def count_vote(txt): return txt.count("vote")
def count_join(txt): return txt.count("join")
def count_send(txt): return txt.count("send")
def count_offer(txt): return txt.count("offer")
def count_deal(txt): return txt.count("deal")
def count_cum(txt): return txt.count("cum")
def count_huge(txt): return txt.count("huge")
def count_from(txt): return txt.count("from")
def count_pill(txt): return txt.count("pill")
def count_hours(txt): return txt.count("hours")
def count_preg(txt): return txt.count("?")
def count_dol(txt): return txt.count("$")
def count_dollar(txt): return txt.count("dollar")
def count_dollars(txt): return txt.count("dollars")
def count_1(txt): return txt.count("1")
def count_2(txt): return txt.count("2")
def count_3(txt): return txt.count("3")
def count_4(txt): return txt.count("4")
def count_5(txt): return txt.count("5")
def count_6(txt): return txt.count("6")
def count_7(txt): return txt.count("7")
def count_8(txt): return txt.count("8")
def count_9(txt): return txt.count("9")
def count_0(txt): return txt.count("0")
def count_work(txt): return txt.count("work")
def count_arr(txt): return txt.count("@")
def count_hash(txt): return txt.count("#")
def count_and(txt): return txt.count("&")
def count_apare(txt): return txt.count("(")
def count_acor(txt): return txt.count("[")
def count_plus(txt): return txt.count("+")
def count_mult(txt): return txt.count("*")
def count_porc(txt): return txt.count("%")
def count_equal(txt): return txt.count("=")
def count_dot(txt): return txt.count(".")
def count_dotc(txt): return txt.count(";")
def count_apos(txt): return txt.count("'")
def count_com(txt): return txt.count("\"")
def count_guionba(txt): return txt.count("_")
def count_dosp(txt): return txt.count(":")
def count_ref(txt): return txt.count("href")
def count_id(txt): return txt.count("id")
def count_px(txt): return txt.count("px")
def count_ESMTP(txt): return txt.count('ESMTP')
def count_menos(txt): return txt.count('-')
def count_sombrero(txt): return txt.count('^')
def count_aparen(txt): return txt.count('(')
def count_cparen(txt): return txt.count(')')
def count_helvetica(txt): return txt.count('helvetica')
def count_arial(txt): return txt.count('arial')
def count_nigeria(txt): return txt.count('nigeria')
def count_win(txt): return txt.count('win')
def count_HTML(txt): return txt.count("HTML")
def count_html(txt): return txt.count("html")
def count_solid(txt): return txt.count("solid;")
def count_microsoft(txt): return txt.count("microsoft")
def count_0600(txt): return txt.count("-0600\nReceived:")
def count_2002(txt): return txt.count("2002")
def count_nahou(txt): return txt.count("nahou-mscnx06p.corp.enron.com")
def count_with(txt): return txt.count("with")
def count_your(txt): return txt.count("your")
def count_0800(txt): return txt.count("-0800\nReceived:")
def count_unv(txt): return txt.count("(unverified)")
def count_sat(txt): return txt.count('Sat,')
def count_sun(txt): return txt.count('Sun,')
def count_upper(txt): return sum([c.isupper() for c in txt])
def count_num(txt): return sum([c.isnumeric() for c in txt])
def count_title(txt): return sum([c.istitle() for c in txt])
def lprom(txt): return sum([len(c) for c in txt])/(len(txt))
def lmax(txt): return max([len(c) for c in txt])

def cargar_mails():
	print 'Cargando ham'
	ham_txt = json.load(open('./dataset_dev_full/ham_dev.json'))
	print 'Cargando spam'
	spam_txt = json.load(open('./dataset_dev_full/spam_dev.json'))

	df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
	df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]
	return df

def guardar_modelo(metodo):
	str_file = metodo + '_entrenado.pickle'	
	fout = open(str_file,'w')
	pickle.dump(clf,fout)
	fout.close()
	return

def cargando_atributos(df):
	dfuncs = [len,count_td,count_by,count_ESMTP,count_menora, count_n,count_upper,count_mm,count_ref,count_guionba
,count_menorp,count_px,count_from,count_id, count_spaces,count_cum,count_viagra,count_sex,count_vagina,count_penis
,count_money,count_earn,count_free,count_now,count_help,count_excl,count_preg,count_dol,count_dollar,count_dollars
,count_1,count_2,count_3,count_4,count_5,count_6,count_7,count_8,count_9,count_0
,count_work,count_arr,count_hash,count_and,count_apare,count_acor,count_plus,count_mult,count_porc,count_equal
,count_dot,count_dotc,count_apos,count_com,count_send,count_menor,count_dosp,count_offer,count_deal,count_join
,count_vote,count_weig,count_lose,count_menos,count_sombrero,count_aparen,count_cparen,count_num,count_title,count_helvetica
,count_arial,count_nigeria,count_win,count_FREE,count_VIAGRA,count_SEX,count_VAGINA,count_PENIS,count_MONEY,count_EARN
,count_NOW,count_html,count_pill,count_hours,lprom,lmax,count_solid,count_font,count_tr,count_microsoft
,count_HTML,count_0600,count_2002,count_nahou,count_with,count_your,count_0800,count_unv,count_sat,count_sun
,count_huge]

	#start_time = time.time()
	print "Cargando atributos"
	for i in range(len(dnames)):
	#	print str(i) + " " + dnames[i]
		df[dnames[i]] = map(dfuncs[i], df.text)
	#	print("--- %s seconds ---" % (time.time() - start_time))
	X = df[dnames].values
	y = df['class']
	return X, y

def leer_base():
	df = cargar_mails()
	train, test = train_test_split(df, test_size = 0.2)
	trainX, trainy = cargando_atributos(train)
	testX, testy = cargando_atributos(test)
	np.save('trainX', trainX)
	np.save('trainy', trainy)
	np.save('testX', testX)
	np.save('testy', testy)

def red_dim(metodo, n):
	trainX = 	np.load('trainX.npy')
	testX = 	np.load('testX.npy')		
	if metodo == "PCA":
		pca = PCA(n_components=n)
		pca.fit(trainX)
		trainX = pca.transform(trainX)
		testX = pca.transform(testX)
	elif metodo == "iPCA":
		pca = IncrementalPCA(n_components=n)
		pca.fit(trainX)
		trainX = pca.transform(trainX)
		testX = pca.transform(testX)
	elif metodo == "ICA":
		ica = FastICA(n_components=n)
		ica.fit(trainX)
		trainX = ica.transform(trainX)
		testX = ica.transform(testX)
	#elif metodo == "RFE":
		#rfe = RFE()
		#rfe.fit(trainX)
		#trainX = rfe.transform(trainX)
		#testX = rfe.transform(testX)
	np.save('trainXt', trainX)
	np.save('testXt', testX)

def predecir(metodo):
	testX = np.load('testX.npy')
	clf = pickle.load(open(metodo + '_entrenado.pickle'))
	predy = list(clf.predict(testX))
	testy = np.load('testy.npy')
	testn = map(lambda x : ( 1 if x == 'ham' else 0 ), testy)
	predn = map(lambda x : ( 1 if x == 'ham' else 0 ), predy)
	print("\tPrecision: %1.3f" % precision_score(testn, predn))
	print("\tRecall: %1.3f" % recall_score(testn, predn))
	print("\tF1: %1.3f" % f1_score(testn, predn))
	print("\tAccuaracy: %1.3f" % accuracy_score(testn, predn))
	print("\tRoc Area Under Curve: %1.3f\n" % roc_auc_score(testn, predn))

if __name__ == '__main__':
	cv = 10
	gs = False
	if len(sys.argv) > 1:
		metodo = sys.argv[1]
		n = 2
		# FILE
		if metodo == 'File':
			leer_base()
			exit()
		# RED
		if metodo == 'Red':
			metodo = "none"
			dims = 10
			if len(sys.argv) > n:
				metodo = sys.argv[n]
				if metodo != "PCA" and metodo != "ICA" and metodo != "RFE" and metodo != "iPCA":
					print u'Dimensión inválida'
					exit()
				n = n + 1
				if len(sys.argv) > n:
					dims = int(sys.argv[n])
					n = n + 1
			red_dim(metodo, dims)
			exit()
		# TEST
		if metodo == 'Test':
			if len(sys.argv) > n:
				metodo = sys.argv[n]
				n = n + 1
			else:
				print u'¿Qué método querés?'
				exit()
			predecir(metodo)
			exit()
		# TRAIN
		if metodo == 'Gsearch':
			if len(sys.argv) > n:
				metodo = sys.argv[n]
			else:
				print u'¿Qué método querés?'
				exit()
			gs = True
			n = n + 1
	else:
		print u'¿Qué método querés?'
		exit()
	if metodo != 'Dtree' and metodo != 'Rforest' and metodo != 'Knn' and metodo != 'Nbayes' and metodo != 'Svc':
		print u'Método inválido'
		exit()
	if len(sys.argv) > n:
		cv = int(sys.argv[n])
		n = n + 1

	X = np.load('trainXt.npy')	
	y = np.load('trainyt.npy')

	if metodo == 'Dtree':
		clf = DecisionTreeClassifier()
		clf.fit(X, y)
	elif metodo == 'Rforest':
		clf = RandomForestClassifier()
		clf.fit(X, y)
	elif metodo == 'Knn':
		clf = KNeighborsClassifier()
		clf.fit(X, y)
	elif metodo == 'Nbayes':
		clf = GaussianNB()
		clf.fit(X, y)
	elif metodo == 'Svc':
		clf = SVC(max_iter=10000)
		clf.fit(X, y)
	
	if (gs):
		param_grid = {}
		if metodo == 'Dtree':
			param_grid = {"max_depth": [5,7,10,15,None],
				"max_features": [10,15,30,50],
				"min_samples_split": [1,3],
				"criterion": ["gini", "entropy"]}
		elif metodo == 'Rforest':
			param_grid = {"max_depth": [1,2],
				"max_features": [10,15],
				"min_samples_split": [1,3],
				"criterion": ["gini", "entropy"]}
		elif metodo == 'Knn':
			param_grid = {"max_depth": [1,2],
				"max_features": [10,15],
				"min_samples_split": [1,3],
				"criterion": ["gini", "entropy"]}
		elif metodo == 'Nbayes':
			param_grid = {"max_depth": [1,2],
				"max_features": [10,15],
				"min_samples_split": [1,3],
				"criterion": ["gini", "entropy"]}
		clf = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,verbose=2)
		X = np.load('trainX.npy')	
		y = np.load('trainy.npy')
		clf.fit(X, y)
		print clf.best_params_
		
		metodo = "GS_" + metodo

	guardar_modelo(metodo)

	res = cross_val_score(clf, X, y, cv=cv)
	print np.mean(res), np.std(res)

#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#-criterion
#-splitter
#-max_features
#-max_depth
#-min_samples_split
#-min_samples_leaf
#-min_weight_fraction_leaf
#-max_leaf_nodes
#-class_weight
#-random_state
#-presort
#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#-n_neighbors
#-weights
#-algorithm
#-leaf_size
#-metric
#-p
#-metric_params
#-n_jobs
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#-n_estimators
#-criterion
#-max_features
#-max_depth
#-min_samples_split
#-min_samples_leaf
#-min_weight_fraction_leaf
#-max_leaf_nodes
#-bootstrap
#-oob_score
#-n_jobs
#-random_state
#-verbose
#-warm_start
#-class_weight
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#-C
#-kernel
#-degree
#-gamma
#-coef0
#-probability
#-shriniking
#-tol
#-cache_size
#-class_weight
#-max_iter
#-decision_function_shape 
#-random_state




