## Loan prediction machine learning project

## Introduction

In finance, a loan is the lending of money by one or more individuals, organizations, or other entities to other individuals, organizations etc. The recipient (i.e., the borrower) incurs a debt and is usually liable to pay interest on that debt until it is repaid as well as to repay the principal amount borrowed. ([wikipedia](https://en.wikipedia.org/wiki/Loan))

Next, we will consider loan prediction based on linear regression.

To do this, the division of the DataSet into training and test sets will be demonstrated. It will be shown how to build models using 3 different machine learning algorithms. Then we will analyze the accuracy and adequacy of the obtained models.

## Objective 
 The main objective for this dataset:[](http://localhost:8888/notebooks/Loan%20Prediction%20Project/loan%20prediction%20machine%20learning%20project.ipynb#The-main-objective-for-this-dataset:)
Using machine learning techniques to predict loan payments.

## Code and Resources used

**Python Version**:3.9.12 

**Packages**:pandas,numpy,sklearn,matplotlib,seaborn

**Data Source**:https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

## Data Collection
The datasets used in this project were downloaded from https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset. One dataset is for the training data and the other one is for the testing data. I then read the two csv files using the pd.read_csv() command.

## Data Cleaning
After downloading the data, I needed to clean it up so that it was usable for our model. I made the following changes
* Removed the Loan_ID columns from both datasets as it is not needed
* Removed columns with the majority of the NaN values
* Replaced the columns with few missing values I replaced the missing values with either the most occuring entry(mode) for categorical data and with the mean value for numeric data. 
* Changed the data types of columns into the correct ones (i.e object for categorical data and float/int for numeric data)

## Exploratory Data Analysis (EDA)
I looked at different distributions for both the numeric and categorical data. Below are highlights from the data visualization section

![bar graph](https://github.com/MusaMasango/loan-prediction-machine-learning-project/blob/main/pivot%20table.png)
![pivot table](https://github.com/MusaMasango/BigData-on-the-spread-of-COVID-19-in-the-world/blob/main/covid%20cases.png)

## Model Building 
The first step of the model building was hypthothesis creation. There are two methods that I used to test my hyphothesis, namely
* Creating models using sklearn
* Time series

First I formulated an hyphothesis based on the number of cases in Africa and the other continents. I then split the data into train and test sets with a test size of 30%. I used the linear regression model then evaluated it using the Mean Absolute Error, Mean Squared Error and Root Mean Squared Error. I then compared the linear regression model with the statsmodel obtained from the statsmodel.api framework. The predicted values from these models are different from the actual values with some uncertainty.

Secondly I used the time series method to test my hyphothesis. In this case we only consider one time series since we are dealing with Africa. I then evaluated it using the Mean Absolute Error, Mean Squared Error and Root Mean Squared Error. The predicted values obtained using the time series are closer to the actual values. 

## Model Performance
Out of the two methods, the time series performed better with an Mean Absolute Error: 765635.4335892488 when compared to the linear regression with an Mean Absolute Error (test): 162489418.69861022
