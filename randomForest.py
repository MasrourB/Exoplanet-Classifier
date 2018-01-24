from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np

np.random.seed(0)

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df['is_train'] = np.random.uniform(0,1,len(df)) <= .75

train, test = df[df['is_train']==True], df[df['is_train']==False]

#print("Num of observations in traing", len(train))
#print("Num of obsercations in testing",len(test))

#print(df.head())

features = df.columns[:4]

#print(features)

y = pd.factorize(train['species'])[0]

clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(train[features], y)

#print(clf.predict(test[features]))

#print(clf.predict_proba(test[features])[0:10])

preds = iris.target_names[clf.predict(test[features])]

#print(preds[0:5])

print(test['species'].head())

#Confusion Matrix
confusion_matrix = pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames= ['Predicted Species'])

print(confusion_matrix)

importance_scores = list(zip(train[features], clf.feature_importances_))

print(importance_scores)
