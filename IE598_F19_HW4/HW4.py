import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read the csv file
df = pd.read_csv("C:\\Users\lyhe\Downloads\housing.csv")

#drop all the Nan in the original data
df =df.dropna()

#makes the plots prettier
sns.set()

#Look at the first five lines and calculate the statistical charactristics for the data
df.head()
df.describe()

#Calculate correlation matrix
corr = df.corr()
print(corr)

#Generate a heatmap
hm = sns.heatmap(corr,cbar=True,annot=False)
hm.set_title('HeatMap of Correlation between the 13 Attributes and MEDV (target value)')
plt.show()

#Scatterplot Matrix of all the features
sns.pairplot(df, height=2.5)
plt.tight_layout()
plt.show()

#Scatterplot Matrix of features with high corr with the target
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], height=2.5)
plt.tight_layout()
plt.show()

#OLS linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Seperate target and features
target = df['MEDV']
features = df.drop(['MEDV'],axis = 1)

#double check the shape of target and features to prevent any mistakes
print(features.shape)
print(target.shape)

#reshape to make sure that it can be taken by the algorithm in sklearn
X = np.array(features).reshape(-1,13)
y = np.array(target).reshape(-1,1)

#train test split, with random seed = 42, test_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2, random_state = 42)

#set alias, fit the data with training set using OLS
reg = LinearRegression()
Reg = reg.fit(X_train, y_train)
y_pred_reg = Reg.predict(X_test)

#Metrics
MSE = metrics.mean_squared_error(y_test, y_pred_reg)
r2 = metrics.r2_score(y_test, y_pred_reg)
print("MSE = " + str(MSE))
print("R^2 = " + str(r2))

REGCOEF = Reg.coef_
labels = df.keys()

for i in range(0,13):
    print("Coef of " + str(labels[i]) + ": " + str(REGCOEF[:,i]))

print("y_intercept = " + str(Reg.intercept_))

RES = y_test - y_pred_reg
RESplot=plt.scatter(range(0,len(y_test)),RES)
plt.title("Residual Graph")
plt.xlabel("# of Y_test")
plt.ylabel("Residual Error")
plt.show()

from sklearn.linear_model import Ridge

alphas = np.logspace(-4,2,7)

rid_mse = list()
for alpha in alphas:
    ridge = Ridge(alpha = alpha)
    rid = ridge.fit(X_train,y_train)
    y_pred = rid.predict(X_test)
    rid_mse.append(metrics.mean_squared_error(y_test, y_pred))

plt.plot(alphas,rid_mse)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title("MSE vs alpha")
plt.show()

alphas = range(0,20)

rid_mse = list()
for alpha in alphas:
    ridge = Ridge(alpha = alpha)
    rid = ridge.fit(X_train,y_train)
    y_pred = rid.predict(X_test)
    rid_mse.append(metrics.mean_squared_error(y_test, y_pred))

plt.plot(alphas,rid_mse)
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title("MSE vs alpha")
plt.show()

ridge = Ridge(alpha = 12)
rid = ridge.fit(X_train,y_train)
y_pred = rid.predict(X_test)
print(metrics.mean_squared_error(y_test, y_pred))
print(metrics.r2_score(y_test, y_pred))

RES = list()
for i in range(0,len(y_test)):
    RES.append(y_test[i,0]-y_pred[i])

RESplot=plt.scatter(range(0,len(y_test)),RES)
plt.title("Residual Graph")
plt.xlabel("# of Y_test")
plt.ylabel("Residual Error")
plt.show()

RIDCOEF = rid.coef_
for i in range(0,13):
    print("Coef of "+str(labels[i])+": "+ str(RIDCOEF[0,i]))

print(rid.intercept_)

from sklearn.linear_model import Lasso
lasso = Lasso()

alphas = np.logspace(-4,2,7)

las_mse= list()
for alpha in alphas:
    lasso = Lasso(alpha = alpha)
    las = lasso.fit(X_train,y_train)
    y_pred = las.predict(X_test)
    las_mse.append(metrics.mean_squared_error(y_test, y_pred))

plt.plot(alphas,las_mse)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title("MSE vs alpha")
plt.show()

alphas = range(0,10)

las_mse= list()
for alpha in alphas:
    lasso = Lasso(alpha = alpha)
    las = lasso.fit(X_train,y_train)
    y_pred = las.predict(X_test)
    las_mse.append(metrics.mean_squared_error(y_test, y_pred))

plt.plot(alphas,las_mse)
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title("MSE vs alpha")
plt.show()

lasso = Lasso(alpha = 1)
las = lasso.fit(X_train,y_train)
y_pred = las.predict(X_test)
print(metrics.mean_squared_error(y_test, y_pred))
print(metrics.r2_score(y_test, y_pred))

RES = list()
for i in range(0,len(y_test)):
    RES.append(y_test[i,0]-y_pred[i])

RESplot=plt.scatter(range(0,len(y_test)),RES)
plt.title("Residual Graph")
plt.xlabel("# of Y_test")
plt.ylabel("Residual Error")
plt.show()

LASCOEF = las.coef_
for i in range(0,13):
    print("Coef of "+str(labels[i])+": "+ str(LASCOEF[i]))

print(lasso.intercept_)
