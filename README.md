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

![bar graph](https://github.com/MusaMasango/loan-prediction-machine-learning-project/blob/main/bar%20graph.png)
![corr plot](https://github.com/MusaMasango/loan-prediction-machine-learning-project/blob/main/correlation%20plot.png)

## Model Building 
First I transformed categorical variables into dummy variables. I also split the data into train and test data sets with a test size of 30%. 

I tried 3 different models and evaluated them using the accuracy score.

The 3 different models used are:
* Logistic regression - I thought I should try something new apart from the linear regression model
* Decision tree - Since we have some categorical data, I thought it would be a good fit
*Random forest - By considering the sparsity associated with the data, I thought it would also be a good fit


## Model Performance
The logistic regression model far outperformed the the other approaches on the test ant validation sets
* Random forest : Accuracy score = 68.11%
* Random forest : Accuracy score = 81.08%
* Logistic regression : Accuracy score = 83.24%

## Conclusion
Credit_History is a very important variable because of its high correlation with Loan_Status therefor showind high Dependancy for the latter.
The Logistic Regression algorithm is the most accurate: approximately 83%
