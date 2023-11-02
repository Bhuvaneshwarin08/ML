import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def kernel(point,xmat,k):
m,n=np.shape(xmat)
weights=np.mat(np.eye((m)))
for j in range(m):
diff=point-x[j]
weights[j,i]=np.exp(diff*diff.T/(2.0*k**2))
return weights
def localweight(point,xmat,ymat,k):
wt=kernel(point,xmat,k)
w=(x.T*(wt*x)).I*(x.T*wt*ymat.T)
return W
def localweight Regression(xmat,ymat,k):

m,n=np.shape(xmat)
ypred=np.zeros(m)
for i in range(m):
ypred[i]=xmat[i]*localweight(xmat[i],xmat,ymat,k)
return ypred
data=pd.read_csv('tips.csv')
col=np.array(data.total_bill)
colB=np.array(data.tip)
mcolA=np.mat(colA)
mcolB=np.mat(colB)
m=np.shape(mcolB)[1]
one=np.ones((1,m),dtype=int)
X=np.hstack((one.T,mcolA.T))
print(X.shape)
ypres=localweightRegression(X,mcolB,0.8)
Xsort=X.copy()
Xsort.sort(axis=0)
plt.scatter(colA,colB,color='blue')
plt.plot(Xsort[:,1],ypred=[x[:,1].argsort(0)],color='yellow',linewidth=5)
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()
