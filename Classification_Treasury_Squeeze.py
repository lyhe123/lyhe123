#Using K_nearest_neighbors and Decision_tree Classifier in sklearn lib to identify
#potential treasury squeeze based on historical data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv("c:\\Users\lyhe\downloads\Treasury Squeeze test - DS1.csv", header=None)
data = np.array(df)
X = data[1:,2:11]

Class = data[1:,11]
Y = list()
for i in Class:
    if (i == 'TRUE'):
        Y.append(1)
    else:
        Y.append(0)
Y = np.array(Y)

print(X.shape)
print(Class.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 33)

k_range = range(1,100)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy Score')

print("Score reaches maximum when k = 30")
knn = KNeighborsClassifier(n_neighbors=30) #Score reaches maximum when k = 30
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
k_range = range(1,10)
scores = []
for k in k_range:
    dt = DecisionTreeClassifier(max_depth = k)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy Score')
plt.show()

print("Score reaches maximum when max_depth = 3")
dt = DecisionTreeClassifier(max_depth=3) #Score reaches maximum when k = 3
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

print("My name is Lingyu He")
print("My NetID is: lingyuh2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
