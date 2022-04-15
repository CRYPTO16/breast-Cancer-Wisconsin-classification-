import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,precision_recall_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv('D:/python dataset for ai/breast-cancer-wisconsin.csv')
dataset=dataset.replace({'?':None})
y=dataset['class']




# correlation matrix for test attribute tells us how much other attributes related to test attribute.
# i.e. in this we can see how much all other attibutes are related to 'class'.
corr_matrix=dataset.corr()
print('correlation metrix for dataset: ',corr_matrix["class"].sort_values(ascending=False))
# because according to correlation matrix id is least related to class.
dataset=dataset.drop(['id'],axis=1)

# visualization of dataset

sns.barplot(x=dataset['class'],y=dataset['cell shape'],color='green')
plt.xlabel('class')
plt.ylabel('cell shape')
plt.show()
sns.barplot(x=dataset['class'],y=dataset['cell size'],color='red')
plt.xlabel('class')
plt.ylabel('cell size')
plt.show()
sns.barplot(x=dataset['class'],y=dataset['bland chromatin'],color='blue')
plt.xlabel('class')
plt.ylabel('bland chromatin')
plt.show()
sns.histplot(x=dataset['class'],y=dataset['mitoses'],color='blue')
plt.xlabel('class')
plt.ylabel('mitoses')
plt.show()

y=dataset['class']
dataset=dataset.drop(['class'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(dataset,y,test_size=0.35)

#model 1    - Acc=0.620833
model1=SGDClassifier(random_state=42)
model1.fit(X_train,y_train)
y_pred=model1.predict(X_test)
print('confusion matrix for classifier: ',confusion_matrix(y_test,y_pred))


#model 2  -Acc=0.691
model2=SVC(kernel='rbf',gamma=5,C=2)
model2.fit(X_train,y_train)
y_pred=model2.predict(X_test)
print('confusion matrix for classifier: ',confusion_matrix(y_test,y_pred))


#model 3   -Acc=0.95
model3=RandomForestClassifier()
model3.fit(X_train,y_train)
y_pred=model3.predict(X_test)
print('confusion matrix for classifier: ',confusion_matrix(y_test,y_pred))

