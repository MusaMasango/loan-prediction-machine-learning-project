## Loan prediction machine learning project

## Introduction

In finance, a loan is the lending of money by one or more individuals, organizations, or other entities to other individuals, organizations etc. The recipient (i.e., the borrower) incurs a debt and is usually liable to pay interest on that debt until it is repaid as well as to repay the principal amount borrowed. ([wikipedia](https://en.wikipedia.org/wiki/Loan))

Next, we will consider loan prediction based on linear regression.

To do this, the division of the DataSet into training and test sets will be demonstrated. It will be shown how to build models using 3 different machine learning algorithms. Then we will analyze the accuracy and adequacy of the obtained models.

## Objective 
The major aim of this project is to predict which of the customers will have their loan approved.

## Stakeholders

The results obtained from this project can be used by various stakeholders within the bank such as
* Credit risk department
* Loan department
* Credit analysts
* Underwriters

## Importance of the project

Manual processing of loan applications is a long, cumbersome, error-prone, and often biased process. It might lead to financial disaster for banks and obstruct genuine applicants from getting the needed loans. Loan Prediction using machine learning tools and techniques can help financial institutions quickly process applications by rejecting high-risk customers entirely, accepting worthy customers, or assigning them to a manual review. Such processes with loan prediction using machine learning intact can reduce loan processing times by nearly 40%.


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
* Logistic regression Classifier 
* Decision tree Classifier
* Random forest Classifier

The reason why I chose this models is beacause since we are dealing with a classification problem these models work best with categorical variables. In addition, these models are easy to implement.

## Model Performance
The logistic regression model far outperformed the the other approaches on the test and validation sets
* Random forest : Accuracy score = 66.67%
* Random forest : Accuracy score = 77.78%
* Logistic regression : Accuracy score = 78.47%

This results makes sense intuitively, since logistic regression algorithm works best where the target variable (dependant variable) is a binary, in this case since the loan status is a binary value between 0 and 1, the logistic regression algorithm will perform better compared to the other models.

## Conclusion
Credit_History is a very important variable because of its high correlation with Loan_Status therefor showind high Dependancy for the latter.
The Logistic Regression algorithm is the most accurate: approximately 78%.
