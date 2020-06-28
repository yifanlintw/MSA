# MSA - Technical Steam

<h3> 1. Import the necessary libraries</h3>
<div>	import pandas as pd </div>	
<div>	import numpy as np</div>	
<div>	from sklearn.tree import DecisionTreeClassifier  </div>	
<div>	from sklearn.model_selection import train_test_split </div>	
<div>	from sklearn import metrics </div>
	
<h3>2. Loading dataset </h3>
<div> iris = pd.read_csv("/Users/Yifanlin/Desktop/Hash Analytic/Assn04/Iris/iris.csv")</div>	

<h3>3.Dividing given columns into two types of variables target variable and feature variables. To understand model performance, dividing the dataset into a training set(70%) and a test set(30%) </h3>
<div> # Split dataset in features and target variable</div>

<div>	iris = iris.drop('Id',axis=1)</div>	
<div>	X = iris.iloc[:, :-1].values</div>	
<div>	y = iris.iloc[:, 4].values </div>
<br><div> # Split dataset into training dataset(70%rows) and test dataset(30% rows)</div>	
<div>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)</div></br>	

<h3> 4.Building  and Evaluating Decision Tree Model</h3>
<div> Creating a Decision Tree Model using Scikit-learn and a decision tree classifer object. Then, train it. </div>
<br><div># Create Decision Tree classifer object</div>	
<div>clf = DecisionTreeClassifier()</div>	
<br>
<div># Train Decision Tree Classifer </div>
<div>clf = clf.fit(X_train,y_train) </div></br>

<div># Predict the response for test dataset </div>
<div>y_pred = clf.predict(X_test) </div></br>

# Question: How often is the classifier correct?
<div>print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) </div>
<div>Accuracy: 0.9555555555555556</div>
<div>The accuracy of the decision tree results is 0.9555555555555556.</div>
<br><div> We can predict the response for test dataset. Based on the model, we can make around 0.9555555555555556 accuracy. </div></br>
