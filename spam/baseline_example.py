# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

# Leo los mails (poner los paths correctos).
ham_txt= json.load(open('./jsons_version/ham_txt.json'))
spam_txt= json.load(open('./jsons_version/spam_txt.json'))

# Imprimo un mail de ham y spam como muestra.
#print ham_txt[0]
print "------------------------------------------------------"
#print spam_txt[0]
print "------------------------------------------------------"

# Armo un dataset de Pandas 
# http://pandas.pydata.org/

df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

# Extraigo dos atributos simples: 
# 1) Longitud del mail.
df['len'] = map(len, df.text)

# 2) Cantidad de espacios en el mail.
def count_spaces(txt): return txt.count(" ")
df['count_spaces'] = map(count_spaces, df.text)

def count_viagra(txt): return txt.count("viagra")
df['count_viagra'] = map(count_viagra, df.text)

def count_sex(txt): return txt.count("sex")
df['count_sex'] = map(count_sex, df.text)

def count_vagina(txt): return txt.count("vagina")
df['count_vagina'] = map(count_vagina, df.text)

def count_penis(txt): return txt.count("penis")
df['count_penis'] = map(count_penis, df.text)

def count_money(txt): return txt.count("money")
df['count_money'] = map(count_money, df.text)

def count_earn(txt): return txt.count("earn")
df['count_earn'] = map(count_earn, df.text)

def count_free(txt): return txt.count("free")
df['count_free'] = map(count_free, df.text)

def count_FREE(txt): return txt.count("FREE")
df['count_FREE'] = map(count_FREE, df.text)

def count_VIAGRA(txt): return txt.count("VIAGRA")
df['count_VIAGRA'] = map(count_VIAGRA, df.text)

def count_SEX(txt): return txt.count("SEX")
df['count_SEX'] = map(count_SEX, df.text)

def count_VAGINA(txt): return txt.count("VAGINA")
df['count_VAGINA'] = map(count_VAGINA, df.text)

def count_PENIS(txt): return txt.count("PENIS")
df['count_PENIS'] = map(count_PENIS, df.text)

def count_MONEY(txt): return txt.count("MONEY")
df['count_MONEY'] = map(count_MONEY, df.text)

def count_EARN(txt): return txt.count("EARN")
df['count_EARN'] = map(count_EARN, df.text)

def count_now(txt): return txt.count("now")
df['count_now'] = map(count_now, df.text)

def count_NOW(txt): return txt.count("NOW")
df['count_NOW'] = map(count_NOW, df.text)

def count_help(txt): return txt.count("help")
df['count_help'] = map(count_help, df.text)

def count_excl(txt): return txt.count("!")
df['count_excl'] = map(count_excl, df.text)

def count_preg(txt): return txt.count("?")
df['count_preg'] = map(count_preg, df.text)

def count_dol(txt): return txt.count("$")
df['count_dol'] = map(count_dol, df.text)

def count_dollar(txt): return txt.count("dollar")
df['count_dollar'] = map(count_dollar, df.text)

def count_dollars(txt): return txt.count("dollars")
df['count_dollars'] = map(count_dollars, df.text)

def count_1(txt): return txt.count("1")
df['count_1'] = map(count_1, df.text)

def count_2(txt): return txt.count("2")
df['count_2'] = map(count_2, df.text)

def count_3(txt): return txt.count("3")
df['count_3'] = map(count_3, df.text)

def count_4(txt): return txt.count("4")
df['count_4'] = map(count_4, df.text)

def count_5(txt): return txt.count("5")
df['count_5'] = map(count_5, df.text)

def count_6(txt): return txt.count("6")
df['count_6'] = map(count_6, df.text)

def count_7(txt): return txt.count("7")
df['count_7'] = map(count_7, df.text)

def count_8(txt): return txt.count("8")
df['count_8'] = map(count_8, df.text)

def count_9(txt): return txt.count("9")
df['count_9'] = map(count_9, df.text)

def count_0(txt): return txt.count("0")
df['count_0'] = map(count_0, df.text)

def count_work(txt): return txt.count("work")
df['count_work'] = map(count_work, df.text)

def count_arr(txt): return txt.count("@")
df['count_arr'] = map(count_arr, df.text)

def count_hash(txt): return txt.count("#")
df['count_hash'] = map(count_hash, df.text)

def count_and(txt): return txt.count("&")
df['count_and'] = map(count_and, df.text)

def count_apare(txt): return txt.count("(")
df['count_apare'] = map(count_apare, df.text)

def count_acor(txt): return txt.count("[")
df['count_acor'] = map(count_acor, df.text)

def count_plus(txt): return txt.count("+")
df['count_plus'] = map(count_plus, df.text)

def count_mult(txt): return txt.count("*")
df['count_mult'] = map(count_mult, df.text)

def count_porc(txt): return txt.count("%")
df['count_porc'] = map(count_porc, df.text)

def count_equal(txt): return txt.count("=")
df['count_equal'] = map(count_equal, df.text)

def count_dot(txt): return txt.count(".")
df['count_dot'] = map(count_dot, df.text)

def count_dotc(txt): return txt.count(";")
df['count_dotc'] = map(count_dotc, df.text)

def count_apos(txt): return txt.count("'")
df['count_apos'] = map(count_apos, df.text)

def count_com(txt): return txt.count("\"")
df['count_com'] = map(count_com, df.text)

def count_send(txt): return txt.count("send")
df['count_send'] = map(count_send, df.text)




#to do: las palabras que se cuentan contarlas primero haciendo lowercase (quizas es mejor preprocesar los jsons haciendo todo lowercase)
# hacer que fr33 cuente como free


# Preparo data para clasificar
X = df[['len', 'count_spaces','count_viagra','count_sex','count_vagina','count_penis','count_money','count_earn','count_free','count_FREE',
'count_VIAGRA','count_SEX','count_VAGINA','count_PENIS','count_EARN','count_MONEY','count_now','count_NOW','count_help',
'count_excl','count_preg','count_dol','count_dollar','count_dollars','count_1','count_2','count_3','count_4','count_5','count_6','count_7'
,'count_8','count_9','count_0','count_work','count_arr','count_hash','count_and','count_apare','count_acor','count_plus','count_mult'
,'count_porc','count_equal','count_dot','count_dotc','count_apos','count_com','count_send']].values
y = df['class']

# Elijo mi clasificador.
clf = DecisionTreeClassifier()

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
res = cross_val_score(clf, X, y, cv=10)
print np.mean(res), np.std(res)
# salida: 0.783040309346 0.0068052434174  (o similar)


