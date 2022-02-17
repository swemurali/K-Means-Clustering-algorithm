# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:

### Step1
Import pandas.

### Step2
Import matplotlib.pyplot.

### Step3
Import sklearn.cluster from KMeans module.

### Step4
Import seaborn

### Step5
Import warnings

### Step6
Declare warnings.filerwarning with ignore as argument

### Step7
Declare a variable x1 and read a csv file(clustering.csv) in it.

### Step8
Declare a variable x2 as index of x1 with arguments ApplicantIncome and LoanAmount.Display x1.head(2) and x2.head(2).

### Step9
Declare a variable x and store x2.values.Declare sns.scatterplot for ApplicantIncome and LoanAmount by indexing.

### Step10
Plot Income , Loan and display them.Declare a variable kmean = KMean(n_cluster_centers_) and execute kmean.fit(x).

### Step11
Display kmean.cluster)centers.Display kmean.labels_

### Step12
Declare a variable predcited_class to kmean.predict([[]]) and give two arguments in it.Display the predicted_class

## Program:
```
##Developed By: M.Suwetha
##Ref.NO:212221230112
 import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
x1 = pd.read_csv('clustering.csv')
print(x1.head(2))
x2 = x1.loc[:,['ApplicantIncome','LoanAmount']]
print(x2.head(2))

x=x2.values
sns.scatterplot(x[:,0],x[:,1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()

kmean = KMeans(n_clusters=4)
kmean.fit(x)

print('Cluster Centers:',kmean.cluster_centers_)
print('Labels:',kmean.labels_)

predicted_class = kmean.predict([[9200,110]])
print('The cluster group for Applicant Income 9000 and Loanamount',predicted_class)
```
## Output:
![output](./ia-6.png)
![output](./ia-6b.png)


## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program.