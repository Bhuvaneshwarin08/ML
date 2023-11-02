import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
col_names=['pregant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima=pd.read_csv("pima_indians_diabetes.csv",header=None,names=col_names)
pima.head()
feature_cols=['pregant','insulin','bmi','age','glucose','bp','pedigree']
x=pima[feature_cols]
y=pima.label
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.neural_network import MLPClassifier
clf=DecisionTreeClassifier()
clf=clf.fix(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
from sklearn.tree import export_graphviz

from six import StringIO
from IPython.display import Image
import pydotplus
dot_data=stringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,special_characters=True,featu
re_names=feature_cols,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getValue())
graph.write_png('diabetes.png')
Image(graph.create_png())
