# MSA - Technical Steam

<h3> 1. Import the necessary libraries</h3>
<h5><div>	import pandas as pd </div>	
<div>	import numpy as np</div>	
<div>	from sklearn.tree import DecisionTreeClassifier  </div>	
<div>	from sklearn.model_selection import train_test_split </div>	
<div>	from sklearn import metrics </div></h5>
	
<h3>2. Loading dataset </h3>
<div> iris = pd.read_csv("/Users/Yifanlin/Desktop/Hash Analytic/Assn04/Iris/iris.csv")</div>	

<h3>3.Dividing given columns into two types of variables target variable and feature variables. To understand model performance, dividing the dataset into a training set(70%) and a test set(30%) is a good strategy. </h3>
<h5> <div> #split dataset in features and target variable</div>	
<div>	iris = iris.drop('Id',axis=1)</div>	
<div>	X = iris.iloc[:, :-1].values</div>	
<div>	y = iris.iloc[:, 4].values </div><h5>

<div>#Split dataset into training dataset(70%rows) and test dataset(30% rows)</div>	
<div>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)</div>	

<h3>4.Building  and Evaluating Decision Tree Model<h3/>
Creating a Decision Tree Model using Scikit-learn. First, creating a decision tree classifer object. Then, train it. Finally, we can predict the response for test dataset. Based on the model, we can make around 0.9555555555555556 accuracy.
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

#Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Question: How often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

Accuracy: 0.9555555555555556
	
The accuracy of the decision tree results is 0.9555555555555556.


What your README should include: 
1.	Detailed description of the idea and project
2.	Environment setup & dependencies 
3.	Step-by-step instruction on how to train and/or test your model (IMPORTANT) 
Further instructions and help will be provided during the Git & GitHub workshop. 
Alternatively please check out this article: 

