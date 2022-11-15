#importing libraries
import pandas as pd     # data processing, CSV file I/O   
import numpy as np      # linear algebra
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns # data visualisation and plotting
import matplotlib.pyplot as plt # data plotting
from sklearn.metrics import accuracy_score
#import dataset
dataset = pd.read_csv('../python/IRIS.csv')
#select X and y values
#X has the 4 features(sepal_lenght/width, petal_lenght/width)
#y has the sample label
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
#label encoding
le = LabelEncoder()
y = le.fit_transform(y)
#testing label encoding
#print(y)
# Adding one more column for bias
rows, col = dataset.shape
X = np.hstack(((np.ones((rows,1))), X))
#data visualization
g = sns.pairplot(dataset, hue='species', markers='X')
plt.show()
#split dataset into training set and test set (80%, 20%)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
train_data = np.column_stack((x_train,y_train)) #combining x and y traning sets
train_data = train_data[train_data[:,5].argsort()] #sorting by species
#print(train_data)
#separate x & y
x_train = train_data[:,0:5]
#print(x_train)
y_train = train_data[:,5]
#print(y_train)

train_rows = len(y_train)
test_rows = len(y_test)
y_train0 = y_train1 = y_train2 = np.array([1]*train_rows)
y_pred = np.array([9]*test_rows)

#Model 1 (class 0-IrisSetosa = 1 while the rest = -1)
for x in range(train_rows):
    if (y_train[x] == 0):
        y_train0[x] = 1
    else:
        y_train0[x] = -1

#print(y_train0)
w0 = np.dot(np.linalg.inv((np.dot(x_train.transpose(),x_train))),(np.dot(x_train.transpose(),y_train0)))
print('y_test = ' + str(y_test)) #Correct values for test sample
#testing Model 1
y_pred0 = np.dot(x_test,w0)
print('y_pred0 = ' + str(y_pred0))
for x in range (test_rows):
    if (y_pred0[x] > 0):
        y_pred[x] = 0
print('Testing First Model')
print('y_pred for class 0 = ' + str(y_pred)) 

#Model 2 (class 1-IrisVersicolor = 1 while the rest = -1)
for x in range(train_rows):
    if (y_train[x] == 1):
        y_train1[x] = 1
    else:
        y_train1[x] = -1

#print(y_train1)
w1 = np.dot(np.linalg.inv((np.dot(x_train.transpose(),x_train))),(np.dot(x_train.transpose(),y_train1)))
#testing Model 2
y_pred1 = np.dot(x_test,w1)
print('y_pred1 = ' + str(y_pred1))
for x in range (test_rows):
    if (y_pred1[x] > 0):
        y_pred[x] = 1
print('Testing Second Model')
print('y_pred for classes 0,1 = ' + str(y_pred))
#Model 3 (class 2-IrisVerginica = 1 while the rest = -1)
for x in range(train_rows):
    if (y_train[x] == 2):
        y_train2[x] = 1
    else:
        y_train2[x] = -1
w2 = np.dot(np.linalg.inv((np.dot(x_train.transpose(),x_train))),(np.dot(x_train.transpose(),y_train2)))
#testing Model 1
y_pred2 = np.dot(x_test,w2)
print('y_pred2 = ' + str(y_pred2))
for x in range (test_rows):
    if (y_pred2[x] > 0):
        y_pred[x] = 2
print('Testing Third Model')
print('y_pred for classes 0, 1, 2 = ' + str(y_pred))

accuracy = accuracy_score(y_test, y_pred)*100
print('Classification accuracy for MSSE is: ' + str(round(accuracy, 2)) + ' %.')
