import csv
import pandas as pd
import string
import emoji
import numpy as np 
from collections import Counter 

import matplotlib.pyplot as plt

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics  import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# Fitting Naive Bayes to the Training set



# read csv file and save it in a data frame
data_df=pd.read_csv('old datasets/tweetsCleanToken_des.csv') 
#data_df=pd.read_csv('TweetsDec.csv') 
# remove data with NAN stance
data_df=data_df[~data_df["class"].isna()]
# idneitfy the data and the labels
data= data_df['T']
target= data_df['class']

# Use TfidfVectorizer for feature extraction (TFIDF to convert textual data to numeric form):
tf_vec = TfidfVectorizer()
X = tf_vec.fit_transform(data)
X.shape
print(tf_vec.vocabulary_)

X_train, X_test, y_train, y_test = train_test_split(X.toarray(), target, test_size=0.40, random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# create the classifer and fit the training data and lables
#classifier_svm = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train,y_train)

classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

print("NB accuracy: %.2f"%classifier_nb.score(X_test, y_test))

#do a 10 fold cross-validation 
results_nb = cross_val_score(classifier_nb, X.toarray(),target, cv=5)
print("\n10-fold cross-validation:")
print(results_nb)

print("The average accuracy of the NB classifier is : %.2f" % np.mean(results_nb))

print("\nConfusion matrix of the NB classifier:")
predicted_nb = classifier_nb.predict(X_test)
print(confusion_matrix(y_test,predicted_nb))


print("\nClassification_report of NB classifier:")
print(classification_report(y_test,predicted_nb))
print("----------------------------------------------------------------------------")
# calculate the fpr and tpr for all thresholds of the classification
probs = classifier_nb.predict_proba(X_test)
preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)


# polt the AUC
plt.title('Receiver Operating Characteristic Naive Bayes classifier')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()