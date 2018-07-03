from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import random		#gia na epilegw tixaia
from random import randrange 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # gia to cosine
from scipy import spatial
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

#auxiliary function which computes the new centroid
def compute_centroid(cluster,X_train):
	aux_matrix=[]
	for index in cluster.keys():
		aux_matrix.append(X_train[index])
	a=np.array(aux_matrix)
	return np.mean(a,axis=0,dtype=np.float64)

#function for the termination of the algorithm,
#if there is no change to the clusters, after the last update on the centroids.

def all_same(old1,old2,old3,old4,old5,new1,new2,new3,new4,new5):
	return (old1==new1) and (old2==new2) and (old3==new3) and (old4==new4) and (old5==new5) 

def print_stats(cluster,politics,business,football,film,technology,i):
	pol1=bus1=foot1=film1=tech1=0
	for key in cluster.keys():
		if cluster[key]=="Politics" :
			pol1+=1
		elif cluster[key]=="Business" : 
			bus1+=1
		elif cluster[key]=="Football" :
			foot1+=1
		elif cluster[key]=="Film" :
			film1+=1
		elif cluster[key]=="Technology" :
			tech1+=1
	import csv
	with open('clustering_KMeans.csv', 'a') as csvfile:
		fieldnames = [' ','Politics','Business','Football','Film','Technology']
		cwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
		cwriter.writerow({' ':'Cluster'+ str(i),'Politics':'%.3f' % (float(pol1)/politics),'Business':'%.3f' % (float(bus1)/business),'Football':'%.3f' % (float(foot1)/football),'Film':'%.3f' % (float(film1)/film),'Technology':'%.3f' % (float(tech1)/technology) })


vectorizer=CountVectorizer(stop_words='english')
df=pd.read_csv("train_set.csv",sep="\t")
svd=TruncatedSVD(n_components=100, random_state=42)
transformer=TfidfTransformer()
pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('svd',svd),
])
X_train=pipeline.fit_transform(df['Content'])

#until here every article has been transformed into a n_components-dimension vector.
centroids=[]
cluster1=cluster2=cluster3=cluster4=cluster5={}
old_cluster1=old_cluster2=old_cluster3=old_cluster4=old_cluster5={}

#random choice of the first 5 centroids.
for i in range(0,5):
	random_index=0
	random_index=randrange(0,len(X_train))
	centroids.append(X_train[random_index])
centroids=np.array(centroids)	

#for each article, find the nearest centroid (via cosine distance) and assign it to the corresponding cluster.
while True:
	old_cluster1=cluster1
	old_cluster2=cluster2
	old_cluster3=cluster3
	old_cluster4=cluster4
	old_cluster5=cluster5
	cluster1={}
	cluster2={}
	cluster3={}
	cluster4={}
	cluster5={}
	for index,row in df.iterrows():
		min_dist=1000000
		to_insert=-1
		for i in range(0,5):
			dist=0.0
			dist=spatial.distance.cosine(centroids[i],X_train[index])
			if dist < min_dist:
				min_dist=dist
				to_insert=i	
		if to_insert==0 :
			cluster1[index]=row["Category"]
		elif to_insert==1 :
			cluster2[index]=row["Category"]
		elif to_insert==2:
			cluster3[index]=row["Category"]
		elif to_insert==3:
			cluster4[index]=row["Category"]
		elif to_insert==4:
			cluster5[index]=row["Category"]
	#compute the new centroids
	centroids[0]=compute_centroid(cluster1,X_train)
	centroids[1]=compute_centroid(cluster2,X_train)
	centroids[2]=compute_centroid(cluster3,X_train)
	centroids[3]=compute_centroid(cluster4,X_train)
	centroids[4]=compute_centroid(cluster5,X_train)
	if all_same(old_cluster1,old_cluster2,old_cluster3,old_cluster4,old_cluster5,cluster1,cluster2,cluster3,cluster4,cluster5):
		break


#count articles for each category from csv
politics=0
business=0
football=0
film=0
technology=0
for index,row in df.iterrows():
	if row["Category"]=="Politics" :
		politics+=1
	elif row["Category"]=="Business":
		business+=1
	elif row["Category"]=="Football":
		football+=1
	elif row["Category"]=="Film":
		film+=1
	elif row["Category"]=="Technology":
		technology+=1

#print the results in a csv file
import csv
with open('clustering_KMeans.csv', 'wb') as csvfile:
	fieldnames = [' ','Politics','Business','Football','Film','Technology']
	cwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
	cwriter.writeheader()
	
	
#print 'Pol: ',politics,' Bus: ',business,' Foot: ',football,' Film: ',film,' Tech: ',technology
print_stats(cluster1,politics,business,football,film,technology,1)
print_stats(cluster2,politics,business,football,film,technology,2)
print_stats(cluster3,politics,business,football,film,technology,3)
print_stats(cluster4,politics,business,football,film,technology,4)
print_stats(cluster5,politics,business,football,film,technology,5)
csvfile.close()




