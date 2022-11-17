#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Output variable -> y
#y -> Whether the client has subscribed a term deposit or not 
#Binomial ("yes" or "no")


# In[2]:


# bank client data:
# # 1 - age (numeric)

# 2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services")

# 3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)

# 4 - education (categorical: "unknown","secondary","primary","tertiary")

# 5 - default: has credit in default? (binary: "yes","no")

# 6 - balance: average yearly balance, in euros (numeric)

# 7 - housing: has housing loan? (binary: "yes","no")

# 8 - loan: has personal loan? (binary: "yes","no")

# related with the last contact of the current campaign:
# 9 - contact: contact communication type (categorical: "unknown","telephone","cellular")

# 10 - day: last contact day of the month (numeric)

# 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")

# 12 - duration: last contact duration, in seconds (numeric)

# other attributes:
# 13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

# 14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)

# 15 - previous: number of contacts performed before this campaign and for this client (numeric)

# 16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success") Output variable (desired target):

# 17 - y - has the client subscribed a term deposit? (binary: "yes","no")

# 18 - Missing Attribute Values: None


# In[3]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[10]:


# Importing the dataset
bank=pd.read_csv(r'C:\Users\anupa\Downloads\bank-full.csv',';')
bank


# In[12]:


#EDA
bank.info()


# In[13]:


# One-Hot Encoding of categrical variables
data1=pd.get_dummies(bank,columns=['job','marital','education','contact','poutcome','month'])
data1


# In[14]:


# To see all columns
pd.set_option("display.max.columns", None)
data1


# In[15]:


data1.info()


# In[16]:


# Custom Binary Encoding of Binary o/p variables 
data1['default'] = np.where(data1['default'].str.contains("yes"), 1, 0)
data1['housing'] = np.where(data1['housing'].str.contains("yes"), 1, 0)
data1['loan'] = np.where(data1['loan'].str.contains("yes"), 1, 0)
data1['y'] = np.where(data1['y'].str.contains("yes"), 1, 0)
data1


# In[17]:


data1.info()


# In[18]:


#Model Building
# Dividing our data into input and output variables
x=pd.concat([data1.iloc[:,0:10],data1.iloc[:,11:]],axis=1)
y=data1.iloc[:,10]


# In[19]:


# Logistic regression model
classifier=LogisticRegression()
classifier.fit(x,y)


# In[20]:


#Model Predictions
# Predict for x dataset
y_pred=classifier.predict(x)
y_pred


# In[21]:


y_pred_df=pd.DataFrame({'actual_y':y,'y_pred_prob':y_pred})
y_pred_df


# In[22]:


#Testing Model Accuracy
# Confusion Matrix for the model accuracy
confusion_matrix = confusion_matrix(y,y_pred)
confusion_matrix


# In[23]:


# The model accuracy is calculated by (a+d)/(a+b+c+d)
(39156+1162)/(39156+766+4127+1162)


# In[24]:


# As accuracy = 0.8933, which is greater than 0.5; Thus [:,1] Threshold value>0.5=1 else [:,0] Threshold value<0.5=0 
classifier.predict_proba(x)[:,1] 


# In[25]:


# ROC Curve plotting and finding AUC value
fpr,tpr,thresholds=roc_curve(y,classifier.predict_proba(x)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(y,y_pred)

plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc)


# In[ ]:




