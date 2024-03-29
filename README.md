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
Our dataset suffers a serious problem of class imbalance. The genuine (not fraud) transactions are more than 99% with the credit card fraud transactions constituting 0.17%.

With such a distribution, if we train our model without taking care of the imbalance issues, it predicts the label with higher importance given to genuine transactions (as there is more data about them) and hence obtains less accuracy.

The class imbalance problem can be solved by various techniques. Oversampling is one of them.

Oversample the minority class is one of the approaches to address the imbalanced datasets. The easiest solution entails doubling examples in the minority class, even though these examples contribute no new data to the model.

Instead, new examples may be generated by replicating existing ones. The Synthetic Minority Oversampling Technique, or SMOTE for short, is a method of data augmentation for the minority class. The SMOTE package using the imblearn library. Now that our dataset was balanced, we proceed with the model building.

We train different models on our dataset and observe which algorithm works better for our problem. This is actually a binary classification problem as we have to predict only 1 of the 2 class labels. We can apply a variety of algorithms for this problem like Random Forest, Decision Tree, Support Vector Machine algorithms, etc.

In this machine learning project, we build 3 classifiers and see which one works best

The 3 different classifiers used are:
* Logistic regression Classifier 
* Decision tree Classifier
* Random forest Classifier


The reason why I chose this algorithms is beacause since we are dealing with a classification problem these models work best with categorical variables. In addition, these models are easy to implement.

## Model Performance
In order to evaluate the model performance for the different classifiers, three classification metrics were used:
* Classification report - Classification report is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model. 

Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).

So, Precision identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.

Mathematically, precision can be defined as the ratio of TP to (TP + FP).


Recall

Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.

Recall identifies the proportion of correctly predicted actual positives.

Mathematically, recall can be given as the ratio of TP to (TP + FN).

f1-score is the weighted harmonic mean of precision and recall. The best possible f1-score would be 1.0 and the worst would be 0.0. f1-score is the harmonic mean of precision and recall. So, f1-score is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of f1-score should be used to compare classifier models, not global accuracy.

Support is the actual number of occurrences of the class in our dataset.

* Confusion matrix - A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-

True Positives (TP) – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.

True Negatives (TN) – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.

False Positives (FP) – False Positives occur when we predict an observation belongs to a certain class but the observation actually does not belong to that class. This type of error is called Type I error.

False Negatives (FN) – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called Type II error.

* Roc curve - ROC Curve stands for Receiver Operating Characteristic Curve. An ROC Curve is a plot which shows the performance of a classification model at various classification threshold levels. The ROC Curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold levels. True Positive Rate (TPR) is also called Recall. It is defined as the ratio of TP to (TP + FN). False Positive Rate (FPR) is defined as the ratio of FP to (FP + TN).

In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR (False Positive Rate) of a single point. This will give us the general performance of the ROC curve which consists of the TPR and FPR at various threshold levels. So, an ROC Curve plots TPR vs FPR at different classification threshold levels. If we lower the threshold levels, it may result in more items being classified as positve. It will increase both True Positives (TP) and False Positives (FP).

ROC AUC stands for Receiver Operating Characteristic - Area Under Curve. It is a technique to compare classifier performance. In this technique, we measure the area under the curve (AUC). A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5, therefore, ROC AUC is the percentage of the ROC plot that is underneath the curve.

Based on the results obtained from these metrics, the Random forest model far outperformed the the other approaches on the test and validation sets as shown below

![model accuracy comparison](https://github.com/MusaMasango/loan-prediction-machine-learning-project/blob/main/model%20accuracy%20comparison.png)

This results makes sense intuitively, since the logistic regression algorithm is faster, hence it will perform better compared to the other algorithms.


## Conclusion
1. In this python machine learning project, I built a binary classifier using the 3 algorithms to predict the loan status. Through this project, I applied techniques to address the loan status imbalance issues and achieved an accuracy of more than 60%. The random forest model yields a very good performance as indicated by the model accuracy which was found to be 78.472222%.
2. Credit_History is a very important variable  because of its high correlation with Loan_Status therefore showing high Dependancy for the latter.
3. To address the issue of loan status imbalance problem, we used the oversampling technique, this was done by the SMOTE package imported from the imblearn module.
4. ROC AUC of our models approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether a loan will be approved or  not.
Depending on the type of dataset, in reality, these models will surely give a competitive performance. In other cases, like the ones where regular payments over a while are a deciding factor, time-series models such as RNNs or LSTMs would perform better.  
This project shows the importance and relevance of using machine learning for loan prediction. We saw some existing approaches and datasets used to approach loan eligibility prediction and how AI might help smoothen this process. Finally, we built an end-to-end loan prediction machine learning project using a publicly available dataset from scratch. At the end of this project, one would know how different features influence the model prediction and how specific attributes affect the decision more than the other features.
