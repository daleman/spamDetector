import os

#print "python tp.py Red PCA 10"
#os.system("python tp.py Red PCA 10")
#print "python tp.py Red iPCA 10"
#os.system("python tp.py Red iPCA 10")
#print "python tp.py Red ICA 10"
#os.system("python tp.py Red ICA 10")
#for i in ["trainX_ICA1.npy","trainX_ICA2.npy","trainX_ICA3.npy","trainX_ICA4.npy","trainX_ICA5.npy","trainX_PCA1.npy","trainX_PCA2.npy","trainX_PCA3.npy","trainX_PCA4.npy","trainX_PCA5.npy","trainX_iPCA1.npy","trainX_iPCA2.npy","trainX_iPCA3.npy","trainX_iPCA4.npy","trainX_iPCA5.npy"]:
for i in ["trainX_ICA10.npy","trainX_PCA10.npy","trainX_iPCA10.npy"]:
  print "DT " + i
  os.system("python tp.py Dtree " + i)
  print "CV DT " + i
  os.system("python tp.py CV Dtree " + i)
  print "RF " + i
  os.system("python tp.py Rforest " + i)
  print "CV RF " + i
  os.system("python tp.py CV Rforest " + i)
  print "NB " + i
  os.system("python tp.py Nbayes " + i)
  print "CV NB " + i
  os.system("python tp.py CV Nbayes " + i)
  print "Knn " + i
  os.system("python tp.py Knn " + i)
  print "CV Knn " + i
#  os.system("python tp.py CV Knn " + i)
#  print "Svc " + i
#  os.system("python tp.py Svc " + i)
#  print "CV Svc " + i
#  os.system("python tp.py CV Svc " + i)
