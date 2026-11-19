import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importing the dataset from my desktop and skipping empty spaces
file_path = '/Users/mybangpetersen/Desktop/iris.csv'
df = pd.read_csv(file_path, skipinitialspace = True)
df.columns = df.columns.str.strip()
#print below to have a look at the numbers
#print(type(df))

#plotting some of the data in a scatterplot
fig, ax = plt.subplots(1, 2, figsize = (15, 5))
ax[0].scatter(
    df['SepalLengthCm'],
    df['SepalWidthCm'],
    #each flower should have its own color
    c = df['Species'].map({
        'Iris-setosa': 'slateblue',
        'Iris-versicolor': 'yellowgreen',
        'Iris-virginica': 'gold',
    }),
)
ax[0].set_xlabel('Sepal Length [cm]')
ax[0].set_ylabel('Sepal Width [cm]')
ax[1].scatter(
    df['PetalLengthCm'],
    df['PetalWidthCm'],
    #each flower should have its own color
    c = df['Species'].map({
        'Iris-setosa': 'slateblue',
        'Iris-versicolor': 'yellowgreen',
        'Iris-virginica': 'gold',
    }),
)
ax[1].set_xlabel('Petal Length [cm]')
ax[1].set_ylabel('Petal Width [cm]')
plt.show()



#by plotting the length of the data sets we see that we have eqal data sets for each specis
fig, ax1 = plt.subplots()
ax1.bar(df['Species'], len(df['PetalLengthCm']),
        color = df['Species'].map({
            'Iris-setosa': 'slateblue',
            'Iris-versicolor': 'yellowgreen',
            'Iris-virginica': 'gold',
        }))
ax1.set_xlabel('Species')
ax1.set_ylabel('PetalLengthCm')
plt.show()

#looking into if the data is normally distributed by plotting a histogram
fig, ax2 = plt.subplots(1, 3, figsize = (15, 5))
ax2[0].hist(df['PetalLengthCm'], bins = 30, color = 'grey')
ax2[0].set_xlabel('Petal Length [cm]')
ax2[1].hist(df['PetalWidthCm'], bins = 30, color = 'grey')
ax2[1].set_xlabel('Petal Width [cm]')
ax2[2].hist(df['SepalLengthCm'], bins = 30, color ='grey')
ax2[2].set_xlabel('Sepal Length [cm]')
plt.show()

#this can also be visualized as a violin plot
plt.figure(figsize = (12, 8))
plt.subplot(2, 2, 1)
sns.violinplot(x = 'Species', y = 'PetalLengthCm', data = df, color = 'grey')
plt.subplot(2, 2, 2)
sns.violinplot(x = 'Species', y = 'PetalWidthCm', data = df, color = 'grey')
plt.subplot(2, 2, 3)
sns.violinplot(x = 'Species', y = 'SepalLengthCm', data = df, color = 'grey')
plt.subplot(2, 2, 4)
sns.violinplot(x = 'Species', y = 'SepalWidthCm', data = df, color = 'grey')
plt.show()  

#importing sklearn packeges for machine learning
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

#we want to predict the species of the iris flower based on the petal measurements
#splitting the data into a training set and a test set, with 25% of the data in the test set
train, test = train_test_split(df, test_size = 0.25)
print('train data:', train.shape)
print('test data:', test.shape)

train_X = train[['PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm', 'SepalWidthCm']]
train_y = train['Species']
test_X = test[['PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm', 'SepalWidthCm']]
test_y = test['Species']

#taking a look at tje data
print('The training data X:')
print(train_X.head())
print('The training data Y')
print(train_y.head())
print('The test data X')
print(test_X.head())
print('The test data Y')
print(test_y.head())

#LogisticRegression gives the probability of the data being in a certain class
model = LogisticRegression()
model.fit(train_X, train_y)
predicted = model.predict(test_X)
print('The predicted values are:', metrics.accuracy_score(predicted, test_y))
#we see that the accuracy is close to 1.0 which means that the model is doing good

#making a confusion matrix to see how the model is doing
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,predicted)
print("Confusion Matrix:")
print(confusion_mat)
print(classification_report(test_y,predicted))
#presicion is the number of true positives divided by the number of true positives plus the number of false positives.
#recall is the number of true positives divided by the number of true positives plus the number of false negatives.
#f1-score is the harmonic mean of precision and recall, which gives a single score that balances both precision and recall.
#support is the number of occurrences of each class in the test set.

#one can examine different statistical models 
#first we look at the nearest nighbor model 
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score NN:",accuracy_score(test_y,y_pred2))

#using support vector machine to classify the data
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score SV:",accuracy_score(test_y,pred_y))

#using decision tree classifier to classify the data
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score DT:",accuracy_score(test_y,y_pred4))

#using gaussian naive bayes to classify the data
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score GN:",accuracy_score(test_y,y_pred3))
#from all the tests we see that the accuracies are in general high 
#the best models are the NN and SV models so we should use them for future reference 







