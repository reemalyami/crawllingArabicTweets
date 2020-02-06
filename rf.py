import csv

import string
import emoji

from collections import Counter 

import matplotlib.pyplot as plt

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics  import confusion_matrix, classification_report
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
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
shape = X.shape
dense = X.todense()
print(X.todense())
print(tf_vec.vocabulary_)

# Random Forest Classifier to the Training set


X_train, X_test, y_train, y_test = train_test_split(X.toarray(), target, test_size=0.20, random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

classifier_rf = RandomForestClassifier(max_depth=70, n_estimators=50, max_features=1)
classifier_rf.fit(X_train, y_train)

print("RF accuracy: %.2f"%classifier_rf.score(X_test, y_test))
#
#do a 10 fold cross-validation
results_rf = cross_val_score(classifier_rf, X.toarray(),target, cv=10)
print("\n10-fold cross-validation:")
print(results_rf)
#
print("The average accuracy of the RF classifier is : %.2f" % np.mean(results_rf))
#
#print("\nConfusion matrix of the RF classifier:")
predicted_rf = classifier_rf.predict(X_test)
print(confusion_matrix(y_test,predicted_rf))
#
#
print("\nClassification_report of RF classifier:")
print(classification_report(y_test,predicted_rf))
print("----------------------------------------------------------------------------")
## calculate the fpr and tpr for all thresholds of the classification
probs = classifier_rf.predict_proba(X_test)
preds = probs[:,1]
#
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
#
#
## polt AUC
plt.title('Receiver Operating Characteristic for Random Forest')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()