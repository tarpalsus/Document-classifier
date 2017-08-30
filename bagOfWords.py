# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:13:56 2017

"""
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn import svm
from sklearn import grid_search
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt


#Data input and manipulation
df=pd.read_csv(r'path_to_cvs.csv')
df2=pd.read_csv(r'path_to_letters.csv')
df3=pd.read_csv(r'path_to_invoices.csv')
df=df.drop('Unnamed: 0',axis=1)
df2=df2.drop('Unnamed: 0',axis=1)
df3=df3.drop('Unnamed: 0',axis=1)
df2['classif']='letter'
df3['classif']='invoice'
df=df.dropna()
df2=df2.dropna()
df3=df3.dropna()
alldf=pd.concat([df,df2,df3])
X=alldf['text']
y=alldf['classif']

def remove_ints(text):
    transformed=[]
    for char in text:
        try:
            char=int(char)
        except:
            transformed.append(char)
    return ''.join(transformed)

#Train/test splitting and vectorization for bag of words
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.4,random_state=7)
X_cross, X_test, y_cross, y_test= train_test_split(X_test,y_test,test_size=0.5,random_state=7)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')

X_train_vector = vectorizer.fit_transform(X_train)

with open('tfidf.pkl', 'wb') as fid:
    pickle.dump(vectorizer, fid)

X=vectorizer.transform(X)
X_test_vector=vectorizer.transform(X_test)
X_cross_vector=vectorizer.transform(X_cross)
feature_names=vectorizer.get_feature_names()

feature_names_no_int=[]
for name in feature_names:
    try:
        word=int(name)
    except:
        feature_names_no_int.append(name)


#Chi square best features selection, not helping
#ch2 = SelectKBest(chi2, k=40)
#X_train = X_train_vector
#X_test= X_test_vector
#X_cross = X_cross_vector
#X_train = ch2.fit_transform(X_train_vector, y_train)
#X_test = ch2.transform(X_test_vector)
#X_cross=ch2.transform(X_cross_vector)
#if feature_names:
#    # keep selected feature names
#    feature_names = [feature_names[i] for i
#                     in ch2.get_support(indices=True)]


#Different model test

modelList=[]
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5, 10]}
model = svm.SVC()
modelList.append({'params':parameters,'model': model})
parameters={}
model = MultinomialNB()
modelList.append({'params':parameters,'model': model})
parameters = {'alpha':[0.5,0.75,1],'binarize':[0]}
model = BernoulliNB()
modelList.append({'params':parameters,'model': model})
model=KNeighborsClassifier()
parameters= {'n_neighbors':list(range(1,10)),'weights':('uniform','distance')}
modelList.append({'params':parameters, 'model' : model})
model=RandomForestClassifier()
parameters= {'n_estimators':list(range(1,10)),'criterion':('gini','entropy'),
             }
modelList.append({'params':parameters, 'model' : model})


for model in modelList:
    classifier = grid_search.GridSearchCV(model['model'], model['params'])
    classifier.fit(X_train, y_train)
    params=classifier.best_params_
    pred = classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print(score,model)

parameters = {'alpha':[0.5,0.75,1],'binarize':[0]}
parameters={}
classifier = grid_search.GridSearchCV(MultinomialNB(), parameters)
classifier.fit(X_train, y_train)

with open('my_dumped_classifier.pkl', 'wb') as fid:
    pickle.dump(classifier, fid)

with open('my_dumped_classifier.pkl', 'rb') as fid:
    classifier_loaded = pickle.load(fid)


#Quick vis
svd=TruncatedSVD(n_components=2)
svd.fit(X)
T=svd.transform(X)
y=y.astype('category').cat.codes
y = np.choose(y, [0,1,2]).astype(np.float)
plt.scatter(T[:,0],T[:,1],c=y)
plt.show()



pred = classifier_loaded.predict(X_test)
print(pred)
score = metrics.accuracy_score(y_test, pred)
print(score,classifier_loaded)