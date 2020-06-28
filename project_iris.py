#!/usr/bin/env python
# coding: utf-8

# In[66]:


# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics # Import scikit-learn metrics module for auuracy calculation
from sklearn.datasets import load_iris


# In[111]:


# load dataset
iris = pd.read_csv("/Users/Yifanlin/Desktop/Iris/iris.csv")


# In[112]:


#split dataset in features and target variable
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

# Split dataset into training dataset(70%rows) and test dataset(30% rows)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 


# In[113]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))




