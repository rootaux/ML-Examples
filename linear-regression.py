####################
# Dataset provided by: Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019
# A simple Machine Learning model based on the Linear Regression which predicts the change of admittance into a foreign university based on the features provided in the dataset.
# This example uses scikit-learn library to train and predict the model and pandas for data handling (I/O csv)
# For check the accuracy of the model we use the R2 method
# Author: Vaibhav Rajput 
####################
import pandas as pd
#Reading the data from csv file using pandas
data = pd.read_csv('admission.csv')
#drop the column 'Serial No.' since it's value does not change the output
data = data.drop('Serial No.', axis=1)

#Check the data for any missing values, it's shape etc
#Uncomment this if you want to check the dataset 
#print(data.shape())
#print(data.head())
#print(data.info()) 
#print(data.describe())

#Import the skikit-learn library
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Split the columns into features and target
x,y = data.loc[:,data.columns != 'Chance of Admit'], data.loc[:,'Chance of Admit']

#Split the dataset into training set and test set
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=1)
#Define the Linear Regression object
regr=LinearRegression()
#Train the model
regr.fit(X_train,y_train)
#Predict for test dataset
prediction = regr.predict(X_test)
#We use R2 method to check the accuracy of the model
print("R2 error:", r2_score(y_test,prediction))

#Example which predicts the chance of admission
GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research=324,107,4,4,4.5,8.87,0
preds = regr.predict([[GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]])  
print('Chance of Admission is', preds[0])
