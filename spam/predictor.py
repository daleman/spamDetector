#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score )
import pickle

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

def cargando_atributos(df):

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

	for i in range(len(dnames)):
		df[dnames[i]] = map(dfuncs[i], df.text)
	return df, dnames

if __name__ == '__main__':
	if len(sys.argv) > 1:
		test = json.load(open(sys.argv[1]))
	else:
		print u'Faltan parámetros!'
		exit()
	metodo = "Dtree"
	df = pd.DataFrame(test, columns=['text'])
	df, dnames = cargando_atributos(df)
	clf = pickle.load(open(metodo + '_entrenado.pickle'))
	predy = list(clf.predict(df[dnames].values))
	for p in predy:
		print p