# MSA - Technical Steam

<h5> 1. Import the necessary libraries<h5/>
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
	
<h5>2. Loading dataset <h5/>
# load dataset
iris = pd.read_csv("/Users/Yifanlin/Desktop/Hash Analytic/Assn04/Iris/iris.csv")

<h5>3.Dividing given columns into two types of variables target variable and feature variables. To understand model performance, dividing the dataset into a training set(70%) and a test set(30%) is a good strategy. <h5/>
#split dataset in features and target variable
iris = iris.drop('Id',axis=1)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

# Split dataset into training dataset(70%rows) and test dataset(30% rows)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

<h3>4.Building  and Evaluating Decision Tree Model<h3/>
Creating a Decision Tree Model using Scikit-learn. First, creating a decision tree classifer object. Then, train it. Finally, we can predict the response for test dataset. Based on the model, we can make around 0.9555555555555556 accuracy.
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

>>>> Accuracy: 0.9555555555555556
	
The accuracy of the decision tree results is 0.9555555555555556


What your README should include: 
1.	Detailed description of the idea and project
2.	Environment setup & dependencies 
3.	Step-by-step instruction on how to train and/or test your model (IMPORTANT) 
Further instructions and help will be provided during the Git & GitHub workshop. 
Alternatively please check out this article: 

