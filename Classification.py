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

def compute_and_print(num,pipeline,X_train,folds):
	
	stats=[]
	fieldnames = ['Statistic Measure', 'Naive Bayes Multinomial', 'Naive Bayes Binomial', 'KNN', 'SVM', 'Random Forest', 'My Method']
	#ROC plot - mean 
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []

	for i, (train_index, test_index) in enumerate(kf):
		#10-fold cross validation (9 samples for training, 1 for testing)
		X_train1, X_test = X_train[train_index], X_train[test_index]
		Y_train1, Y_test = Y_train[train_index], Y_train[test_index]

		probas_ = pipeline.fit(X_train1,Y_train1).predict(X_test)
		
		if i == 0:
			yar = Y_test
			pr = probas_
		elif i == 1:
			yar1 = Y_test
			pr1 = probas_
		elif i == 2:
			yar2 = Y_test
			pr2 = probas_
		elif i == 3:
			yar3 = Y_test
			pr3 = probas_
		elif i == 4:
			yar4 = Y_test
			pr4 = probas_
		elif i == 5:
			yar5 = Y_test
			pr5 = probas_
		elif i == 6:
			yar6 = Y_test
			pr6 = probas_
		elif i == 7:
			yar7 = Y_test
			pr7 = probas_
		elif i == 8:
			yar8 = Y_test
			pr8 = probas_
		elif i == 9:
			yar9 = Y_test
			pr9 = probas_

		# Accuracy
		stats.append(accuracy_score(Y_test, probas_))
	#ROC plot 
	for j in range(5):
		mean_tpr = 0.0
		mean_fpr = np.linspace(0, 1, 100)
		all_tpr = []
		

		for i in range(10):
			# Compute ROC curve and area the curve
			if i == 0:
				fpr, tpr, thresholds = roc_curve(yar, pr, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))
			elif i == 1:
				fpr, tpr, thresholds = roc_curve(yar1, pr1, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))
			elif i == 2:
				fpr, tpr, thresholds = roc_curve(yar2, pr2, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))
			elif i == 3:
				fpr, tpr, thresholds = roc_curve(yar3, pr3, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))
			elif i == 4:
				fpr, tpr, thresholds = roc_curve(yar4, pr4, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))
			elif i == 5:
				fpr, tpr, thresholds = roc_curve(yar5, pr5, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))
			elif i == 6:
				fpr, tpr, thresholds = roc_curve(yar6, pr6, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))
			elif i == 7:
				fpr, tpr, thresholds = roc_curve(yar7, pr7, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))
			elif i == 8:
				fpr, tpr, thresholds = roc_curve(yar8, pr8, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))
			elif i == 9:
				fpr, tpr, thresholds = roc_curve(yar9, pr9, pos_label = j) # pos_label is for the 5 categories
				mean_tpr += interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (i, roc_auc))

		plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
		mean_tpr /= len(kf)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		stats.append(mean_auc)
		plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
		plt.xlim([0.0, 1.05])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

		if num==1:
			plt.title('MultinomialNB'+'-'+le.classes_[j])
			lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.savefig(fieldnames[1]+'_'+le.classes_[j]+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
			plt.clf()
		elif num==2:
			plt.title('Binomial'+'-'+le.classes_[j])
			lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.savefig(fieldnames[2]+'_'+le.classes_[j]+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
			plt.clf()
		elif num==3:
			plt.title('KNN'+'-'+le.classes_[j])
			lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.savefig(fieldnames[3]+'_'+le.classes_[j]+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
			plt.clf()
		elif num==4:
			plt.title('SVM'+'-'+le.classes_[j])
			lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.savefig(fieldnames[4]+'_'+le.classes_[j]+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
			plt.clf()
		elif num==5:
			plt.title('RandomForest'+'-'+le.classes_[j])
			lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.savefig(fieldnames[5]+'_'+le.classes_[j]+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
			plt.clf()
		elif num==6:
			plt.title('MyMethod'+'-'+le.classes_[j])
			lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.savefig(fieldnames[6]+'_'+le.classes_[j]+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
			plt.clf()
	return stats

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

vectorizer=CountVectorizer(stop_words='english')
transformer=TfidfTransformer()
svd=TruncatedSVD(n_components=200, random_state=42) 

stats = ['Accuracy','','','','','','','','','','ROC','','','','']
stats1 = []
stats2 = []
stats3 = []
stats4 = []
stats5 = []
stats6 = []
fieldnames = ['Statistic Measure', 'Naive Bayes Multinomial', 'Naive Bayes Binomial', 'KNN', 'SVM', 'Random Forest', 'My Method']
import csv
csv_out = open('EvaluationMetric_10fold.csv', 'wb')
clwriter = csv.writer(csv_out)

kf = KFold(len(X_train), n_folds=10) 

#Multinomial NaiveBayes
clf=MultinomialNB()

pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('clf', clf)
])

stats1=compute_and_print(1,pipeline,X_train,kf)


#bernouli
clf=BernoulliNB()
pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('svd',svd),
    ('clf', clf)
])
stats2=compute_and_print(2,pipeline,X_train,kf)

#knn
clf=KNeighborsClassifier(n_jobs=-1)
pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('svd',svd),
    ('clf', clf)
])
stats3=compute_and_print(3,pipeline,X_train,kf)

#svm
clf=svm.SVC()
pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('svd',svd),
    ('clf', clf)
])
stats4=compute_and_print(4,pipeline,X_train,kf)

#randomforest
clf=RandomForestClassifier(n_jobs=-1)
pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('svd',svd),
    ('clf', clf)
])
stats5=compute_and_print(5,pipeline,X_train,kf)

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

stats6=compute_and_print(6,pipeline,X_train,kf)

#save stats to csv file

rows = zip(stats, stats1, stats2, stats3, stats4, stats5, stats6)
clwriter.writerow(fieldnames)
clwriter.writerows(rows)
csv_out.close()

