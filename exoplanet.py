from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

np.random.seed(0)

#f = open("kepler-labelled-time-series-data/exoTest.csv")

#Pandas df
train = pd.read_csv("kepler-labelled-time-series-data/exoTrain.csv")
test = pd.read_csv("kepler-labelled-time-series-data/exoTest.csv")

#print(df.head())
features = train.columns[1:30]

train_x = train[features]
test_x = test[features]

train_y = train.LABEL
test_y = test.LABEL

train_x, train_y = shuffle(train_x, train_y)
#print(train_labels)
clf_gnb = GaussianNB()

clf_log_reg = LogisticRegression()

clf_svm = svm.SVC(kernel='linear', C=1)

clf_rand_forest = RandomForestClassifier(n_jobs=2, random_state=0)

k_fold = KFold(n_splits=10)
print("Cross Val Score for GNB", cross_val_score(clf_gnb, train_x, train_y, cv=k_fold, n_jobs=-1))
print("Cross Val Score for Log Reg", cross_val_score(clf_log_reg, train_x, train_y, cv=k_fold, n_jobs=-1))
print("Cross Val Score for SVM", cross_val_score(clf_svm, train_x, train_y, cv=k_fold, n_jobs=-1))
print("Cross Val Score for Random Forest", cross_val_score(clf_rand_forest, train_x, train_y, cv=k_fold, n_jobs=-1))

clf_rand_forest.fit(train_x, train_y)


#print("Naive Bayes train accuracy", accuracy_score(train_y, clf_gnb.predict(train_x)))
#print("Naive Bayes test accuracy", accuracy_score(test_y, clf_gnb.predict(test_x)))

#print("Log Reg train accuracy", accuracy_score(train_y, clf_log_reg.predict(train_x)))
#print("Log Reg test accuracy", accuracy_score(test_y, clf_log_reg.predict(test_x)))

#print("SVM train accuracy", accuracy_score(train_y, clf_svm.predict(train_x)))
#print("SVM test accuracy", accuracy_score(test_y, clf_svm.predict(test_x)))

print("Train accuracy", accuracy_score(train_y, clf_rand_forest.predict(train_x)))
print("Test accuray", accuracy_score(test_y, clf_rand_forest.predict(test_x)))

#Confusion Matrix
print(list(zip(train_x, clf_rand_forest.feature_importances_)))
