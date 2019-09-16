#Modified From Bowles - Chapter 2

import numpy as np
import pandas as pd

xList = []

labels = []

df.head()

file = open("c:\\Users\lyhe\downloads\HY_Universe_corporate bond.csv")

for line in file:
    row = line.strip().split(",")
    xList.append(row)

import sys

sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])))

nrow = len(xList)
ncol = len(xList[1])

type = [0]*3
colCounts = []

for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
        else:
                type[2] += 1

    colCounts.append(type)
    type = [0]*3

sys.stdout.write("Col#" + '\t\t' + "Number" + '\t\t' + "Strings" + '\t\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' + str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1

col = 28
colData = []
for row in xList:
    colData.append(float(row[col]))

col = 27
colData = []
for row in xList[1:]:
    colData.append(float(row[col]))

colArray = np.array(colData)

colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' + "Standard Deviation = " + '\t ' + str(colsd) + "\n")

ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))

sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

ntiles = 10
percentBdry = []

for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))

sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

col = 14
colData = []

for row in xList[1:]:
    colData.append(row[col])

unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)

catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*len(unique)

for elt in colData:
    catCount[catDict[elt]] += 1

sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)

import pylab
import scipy.stats as stats
col = 27
colData = []
for row in xList[1:]:
    colData.append(float(row[col]))

stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()

df = pd.read_csv("c:\\Users\lyhe\downloads\HY_Universe_corporate bond.csv", header=None)

df.head()

df.tail()

df.describe()

from pandas import DataFrame
import matplotlib.pyplot as plot

att2 = df.iloc[1:50,20]
att3 = df.iloc[1:50,21]

plot.scatter(att2, att3)
plot.xlabel("LIQ SCORE")
plot.ylabel(("Number of Trades"))
plot.show()

att4 = df.iloc[1:50,22]

plot.scatter(att2, att4)
plot.xlabel("LIQ SCORE")
plot.ylabel(("Volume"))
plot.show()
