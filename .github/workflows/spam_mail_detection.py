import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction_text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import SVM
data=pd.read_csv('spam.csv')
data.head()
data.info()
X=data['message'].values
Y=data['category'].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
CV=CountVectorizer()
X_train=CV.fit_transform(X_train)
X_test=CV.transform(X_test)
from sklearn.svm import SVC

classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,Y_train)
print(classifier.score(X_test,Y_test)
