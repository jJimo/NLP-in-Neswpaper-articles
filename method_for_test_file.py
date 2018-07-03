from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#k-fold cross validation
from sklearn.cross_validation import KFold
#accuracy
from sklearn.metrics import accuracy_score

#roc
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp

# my method
from sklearn.ensemble import VotingClassifier
#Read Data
df=pd.read_csv("train_set.csv",sep="\t")
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train=le.transform(df["Category"])
X_train1=df['Content'] 
X_train2=[]
for i in range(len(X_train1)):
	X_train2.append(10*df['Title'][i]+df['Content'][i])

X_train=np.array(X_train2)

#read test file
df_test=pd.read_csv("test_set.csv",sep="\t")

vectorizer=CountVectorizer(stop_words='english')
transformer=TfidfTransformer()
svd=TruncatedSVD(n_components=200, random_state=42) 
pipeline_test = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('svd',svd),
])
#My method---Voting Classifier
clf1 = BernoulliNB(fit_prior=False)
clf2 = KNeighborsClassifier(weights='distance',n_jobs=-1)
clf3 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
clf = VotingClassifier(estimators=[('bnb',clf1),('knn',clf2),('rf',clf3)], voting='hard')
pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('svd',svd),
    ('clf', clf)
])
pipeline.fit(X_train,Y_train)

#let's predict the test
X_test=pipeline_test.fit_transform(df_test['Content'])

predicted=clf.predict(X_test)
ids=[]
pred_cat=[]
for i in range(len(X_test)):
	ids.append(df_test['Id'][i])
	pred_cat.append(le.classes_[predicted[i]])

fieldnames=['ID','Predicted_Category']

import csv
csv_out = open('testSet_categories.csv', 'wb')
clwriter = csv.writer(csv_out)
rows = zip(ids, pred_cat)
clwriter.writerow(fieldnames)
clwriter.writerows(rows)
csv_out.close()

