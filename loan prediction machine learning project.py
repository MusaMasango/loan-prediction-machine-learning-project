#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# <table>
#   <tr><td>
#     <img src="https://pas-wordpress-media.s3.us-east-1.amazonaws.com/content/uploads/2015/12/loan-e1450497559334.jpg"
#          width="400" height="600">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# In finance, a loan is the lending of money by one or more individuals, organizations, or other entities to other individuals, organizations etc. The recipient (i.e., the borrower) incurs a debt and is usually liable to pay interest on that debt until it is repaid as well as to repay the principal amount borrowed. ([wikipedia](https://en.wikipedia.org/wiki/Loan))
# 
# ### **The major aim of this notebook is to predict which of the customers will have their loan approved.**
# 
# ![](https://i.pinimg.com/originals/41/b0/08/41b008395e8e7f888666688915750d1f.gif)
# 
# # Data Id 📋
# 
# This dataset is named [Loan Prediction Dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset) data set. The dataset contains a set of **613** records under **13 attributes**:
# 
# ![](http://miro.medium.com/max/795/1*cAd_tqzgCWtCVMjEasWmpQ.png)
# 
# ## The main objective for this dataset:
# Using machine learning techniques to predict loan payments.
# 
# ### target value: `Loan_Status`
# 
# # Libraries 📕📗📘

# In[4]:


import os #paths to file
import numpy as np # numpy library
import pandas as pd # data processing
import warnings# warning filter


#ploting libraries
import matplotlib.pyplot as plt 
import seaborn as sns

#relevant ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#default theme
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

#warning hadle
warnings.filterwarnings("ignore")


# In[49]:


file = open("C:/Users/Musa Masango/Downloads/test_Y3wMUE5_7gLdaTN.xls", "r")
content = file.read()
print(content)
file.close()


# In[56]:


#path for the training set
train_path = "C:/Users/Musa Masango/Downloads/train_u6lujuX_CVtuZ9i.csv"
#path for the testing set
test_path = "C:/Users/Musa Masango/Downloads/test_Y3wMUE5_7gLdaTN.xls"


# # Preprocessing and Data Analysis 💻
# 
# ## First look at the data:
# 
# Training set:

# In[54]:


# read in csv file as a DataFrame

train_df = pd.read_csv(tr_path)
# explore the first 5 rows
train_df.head()


# Testing set:

# In[58]:


# read in csv file as a DataFrame
test_df = pd.read_csv(test_path)
# explore the first 5 rows
test_df.head()


# Size of each data set:

# In[59]:


print(f"training set (row, col): {train_df.shape}\n\ntesting set (row, col): {test_df.shape}")


# ### Now the focus is shifted for the preprocessing of the training dataset.

# In[60]:


#column information
train_df.info(verbose=True, null_counts=True)


# In[61]:


#summary statistics
train_df.describe()


# In[63]:


#the Id column is not needed, let's drop it for both test and train datasets
train_df.drop('Loan_ID',axis=1,inplace=True)
test_df.drop('Loan_ID',axis=1,inplace=True)
#checking the new shapes
print(f"training set (row, col): {train_df.shape}\n\ntesting set (row, col): {test_df.shape}")


# ## Missing values 🚫
# As you can see we have some missing data, let's have a look how many we have for each column:

# In[64]:


#missing values in decsending order
train_df.isnull().sum().sort_values(ascending=False)


# Each value will be replaced by the most frequent value (mode).
# 
# E.G. `Credit_History` has 50 null values and has 2 unique values `1.0` (475 times) or `0.0` (89 times) therefore each null value will be replaced by the mode `1.0` so now it will show in our data 525 times. 

# In[65]:


#filling the missing data
print("Before filling missing values\n\n","#"*50,"\n")
null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']


for col in null_cols:
    print(f"{col}:\n{train_df[col].value_counts()}\n","-"*50)
    train_df[col] = train_df[col].fillna(
    train_df[col].dropna().mode().values[0] )   

    
train_df.isnull().sum().sort_values(ascending=False)
print("After filling missing values\n\n","#"*50,"\n")
for col in null_cols:
    print(f"\n{col}:\n{train_df[col].value_counts()}\n","-"*50)


# # Data visualization 📊

# Firstly we need to split our data to categorical and numerical data,
# 
# 
# using the `.select_dtypes('dtype').columns.to_list()` combination.

# ## Loan status distribution

# In[66]:


#list of all the columns.columns
#Cols = tr_df.tolist()
#list of all the numeric columns
num = train_df.select_dtypes('number').columns.to_list()
#list of all the categoric columns
cat = train_df.select_dtypes('object').columns.to_list()

#numeric df
loan_num =  train_df[num]
#categoric df
loan_cat = train_df[cat]


# In[67]:


print(train_df[cat[-1]].value_counts())
#tr_df[cat[-1]].hist(grid = False)

#print(i)
total = float(len(train_df[cat[-1]]))
plt.figure(figsize=(8,10))
sns.set(style="whitegrid")
ax = sns.countplot(train_df[cat[-1]])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center") 
plt.show()


# Let's plot our data
# 
# Numeric:

# In[68]:


for i in loan_num:
    plt.hist(loan_num[i])
    plt.title(i)
    plt.show()


# Categorical (split by Loan status):

# In[70]:


for i in cat[:-1]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i ,hue='Loan_Status', data=train_df ,palette='plasma')
    plt.xlabel(i, fontsize=14)


# ## Encoding data to numeric

# In[72]:


#converting categorical values to numbers

to_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}

# adding the new numeric values from the to_numeric variable to both datasets
train_df = train_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
test_df = test_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)

# convertind the Dependents column
Dependents_ = pd.to_numeric(train_df.Dependents)
Dependents__ = pd.to_numeric(test_df.Dependents)

# dropping the previous Dependents column
train_df.drop(['Dependents'], axis = 1, inplace = True)
test_df.drop(['Dependents'], axis = 1, inplace = True)

# concatination of the new Dependents column with both datasets
train_df = pd.concat([train_df, Dependents_], axis = 1)
test_df = pd.concat([test_df, Dependents__], axis = 1)

# checking the our manipulated dataset for validation
print(f"training set (row, col): {train_df.shape}\n\ntesting set (row, col): {test_df.shape}\n")
print(train_df.info(), "\n\n", test_df.info())


# ## Correlation matrix 

# In[73]:


#plotting the correlation matrix
sns.heatmap(train_df.corr() ,cmap='cubehelix_r')


# ### Correlation table for a more detailed analysis:

# In[74]:


#correlation table
corr = train_df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# We can clearly see that `Credit_History` has the highest correlation with `Loan_Status` (a positive correlation of `0.54`).
# Therefore our target value is highly dependant on this column.

# # Machine learning models
# 
# First of all we will divide our dataset into two variables `X` as the features we defined earlier and `y` as the `Loan_Status` the target value we want to predict.
# 
# ## Models we will use:
# 
# * **Decision Tree** 
# * **Random Forest**
# * **Logistic Regression**
# 
# ## The Process of Modeling the Data:
# 
# 1. Importing the model
# 
# 2. Fitting the model
# 
# 3. Predicting Loan Status
# 
# 4. Classification report by Loan Status
# 
# 5. Overall accuracy
# 

# In[75]:


y = train_df['Loan_Status']
X = train_df.drop('Loan_Status', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# ## Decision Tree
# 
# ![](https://i.pinimg.com/originals/eb/08/05/eb0805eb6e34bf3eac5ab4666bbcc167.gif)

# In[76]:


DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

y_predict = DT.predict(X_test)

#  prediction Summary by species
print(classification_report(y_test, y_predict))

# Accuracy score
DT_SC = accuracy_score(y_predict,y_test)
print(f"{round(DT_SC*100,2)}% Accurate")


# ### Csv results of the test for our model:
# 
# <table>
#   <tr><td>
#     <img src="https://miro.medium.com/max/900/1*a99bY1VkmfXhqW-5uAX28w.jpeg"
#          width="200" height="300">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# You can see each predition and true value side by side by the csv created in the output directory.

# In[77]:


Decision_Tree=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Decision_Tree.to_csv("Dection Tree.csv")     


# ## Random Forest
# 
# ![](https://miro.medium.com/max/1280/1*9kACduxnce_JdTrftM_bsA.gif)

# In[78]:


RF = RandomForestClassifier()
RF.fit(X_train, y_train)

y_predict = RF.predict(X_test)

#  prediction Summary by species
print(classification_report(y_test, y_predict))

# Accuracy score
RF_SC = accuracy_score(y_predict,y_test)
print(f"{round(RF_SC*100,2)}% Accurate")


# ### Csv results of the test for our model:
# 
# <table>
#   <tr><td>
#     <img src="https://miro.medium.com/max/900/1*a99bY1VkmfXhqW-5uAX28w.jpeg"
#          width="200" height="300">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# You can see each predition and true value side by side by the csv created in the output directory.

# In[80]:


Random_Forest=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Random_Forest.to_csv("Random Forest.csv")     


# ### Csv results of the test for our model:
# 
# <table>
#   <tr><td>
#     <img src="https://miro.medium.com/max/900/1*a99bY1VkmfXhqW-5uAX28w.jpeg"
#          width="200" height="300">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# You can see each predition and true value side by side by the csv created in the output directory.

# ## Logistic Regression
# Now, I will explore the Logistic Regression model.
# 
# <table>
#   <tr><td>
#     <img src="https://files.realpython.com/media/log-reg-2.e88a21607ba3.png"
#           width="500" height="400">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>

# In[83]:


LR = LogisticRegression()
LR.fit(X_train, y_train)

y_predict = LR.predict(X_test)

#  prediction Summary by species
print(classification_report(y_test, y_predict))

# Accuracy score
LR_SC = accuracy_score(y_predict,y_test)
print(f"{round(LR_SC*100,2)}% Accurate")


# In[27]:


Logistic_Regression=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Logistic_Regression.to_csv("Logistic Regression.csv")     


# ### Csv results of the test for our model:
# 
# <table>
#   <tr><td>
#     <img src="https://miro.medium.com/max/900/1*a99bY1VkmfXhqW-5uAX28w.jpeg"
#          width="200" height="300">
#       <tr><td align="center">
#   </td></tr>
#   </td></tr>
# </table>
# 
# You can see each predition and true value side by side by the csv created in the output directory.

# # Conclusion
# 
# 1. `Credit_History` is a very important variable  because of its high correlation with `Loan_Status` therefor showind high Dependancy for the latter.
# 2. The Logistic Regression algorithm is the most accurate: **approximately 83%**.

# In[85]:


score = [DT_SC,RF_SC,LR_SC]
Models = pd.DataFrame({
    'n_neighbors': ["Decision Tree","Random Forest", "Logistic Regression"],
    'Score': score})
Models.sort_values(by='Score', ascending=False)

