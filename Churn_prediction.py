# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:55:55 2018

@author: mohammad.sharibkhan
"""

import pandas as pd


import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("D:\\D-Drive Backup\\Python\\Telecom Churn\\WA_FnUseC_TelcoCustomerChurn.csv")

#  drop customer Id
df=df.drop(['customerID'],axis=1)

#  drop gender
df=df.drop(['gender'],axis=1)

# hot and label encoder on contarct
Contract = pd.get_dummies(df['Contract'],drop_first=True, prefix='contract')
df=df.drop(['Contract'],axis=1)

df = pd.concat([df,Contract],axis=1)

# hot and label encoder on SeniorCitizen
SeniorCitizen = pd.get_dummies(df['SeniorCitizen'],drop_first=True, prefix='SeniorCitizen')
df=df.drop(['SeniorCitizen'],axis=1)

df = pd.concat([df,SeniorCitizen],axis=1)


# hot and label encoder on SeniorCitizen
#MultipleLines = pd.get_dummies(df['MultipleLines'],drop_first=True, prefix='MultipleLines')
#df=df.drop(['MultipleLines'],axis=1)

#df = pd.concat([df,MultipleLines],axis=1)

# hot and label encoder on MultipleLines
InternetService = pd.get_dummies(df['InternetService'],drop_first=True, prefix='InternetService')
df=df.drop(['InternetService'],axis=1)

df = pd.concat([df,InternetService],axis=1)

# hot and label encoder on PhoneService
#PhoneService = pd.get_dummies(df['PhoneService'],drop_first=True, prefix='PhoneService')
df=df.drop(['PhoneService'],axis=1)
df=df.drop(['MultipleLines'],axis=1)


#df = pd.concat([df,PhoneService],axis=1)

# hot and label encoder on Dependent
Dependents = pd.get_dummies(df['Dependents'],drop_first=True, prefix='Dependents')
df=df.drop(['Dependents'],axis=1)

df = pd.concat([df,Dependents],axis=1)

# hot and label encoder on Partners
Partner = pd.get_dummies(df['Partner'],drop_first=True, prefix='Partner')
df=df.drop(['Partner'],axis=1)

df = pd.concat([df,Partner],axis=1)

# hot and label encoder on OnlineSecurity
OnlineSecurity = pd.get_dummies(df['OnlineSecurity'],drop_first=True, prefix='OnlineSecurity')
df=df.drop(['OnlineSecurity'],axis=1)

df = pd.concat([df,OnlineSecurity],axis=1)

# hot and label encoder on OnlineBackup
OnlineBackup = pd.get_dummies(df['OnlineBackup'],drop_first=True, prefix='OnlineBackup')
df=df.drop(['OnlineBackup'],axis=1)

df = pd.concat([df,OnlineBackup],axis=1)

# hot and label encoder on DeviceProtection
DeviceProtection = pd.get_dummies(df['DeviceProtection'],drop_first=True, prefix='DeviceProtection')
df=df.drop(['DeviceProtection'],axis=1)

df = pd.concat([df,DeviceProtection],axis=1)

# hot and label encoder on TechSupport
TechSupport = pd.get_dummies(df['TechSupport'],drop_first=True, prefix='TechSupport')
df=df.drop(['TechSupport'],axis=1)

df = pd.concat([df,TechSupport],axis=1)

# hot and label encoder on TechSupport
StreamingTV = pd.get_dummies(df['StreamingTV'],drop_first=True, prefix='StreamingTV')
df=df.drop(['StreamingTV'],axis=1)

df = pd.concat([df,StreamingTV],axis=1)

# hot and label encoder on TechSupport
StreamingMovies = pd.get_dummies(df['StreamingMovies'],drop_first=True, prefix='StreamingMovies')
df=df.drop(['StreamingMovies'],axis=1)

df = pd.concat([df,StreamingMovies],axis=1)

# hot and label encoder on PaperlessBilling
PaperlessBilling = pd.get_dummies(df['PaperlessBilling'],drop_first=True, prefix='PaperlessBilling')
df=df.drop(['PaperlessBilling'],axis=1)

df = pd.concat([df,PaperlessBilling],axis=1)

# hot and label encoder on PaymentMethod
PaymentMethod = pd.get_dummies(df['PaymentMethod'],drop_first=True, prefix='PaymentMethod')
df=df.drop(['PaymentMethod'],axis=1)

df = pd.concat([df,PaymentMethod],axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.dropna(axis=0, inplace=True)

from sklearn.cross_validation import train_test_split

X=df

#df['Churn']=df['Churn'].replace(['NO'], '1')
df['Churn']=df['Churn'].map({'Yes': 0, 'No': 1})
Y=df['Churn']
#X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
X=X.drop(['Churn'],axis=1)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.cross_validation import cross_val_score

from xgboost import XGBClassifier
XGBmodel = XGBClassifier()

XGBScore=cross_val_score(XGBmodel,X_train,Y_train,cv=10 )
XGBScore.mean()

XGBmodel.fit(X_train,Y_train)
XGBY_pred = XGBmodel.predict(X_test)
XGBmodel.score(X_train,Y_train)
XGBmodel.score(X_test,Y_test)

from sklearn.ensemble import RandomForestClassifier
RFModel = RandomForestClassifier(n_estimators = 100)

RFscore=cross_val_score(RFModel,X_train,Y_train,cv=10 )
RFscore.mean()

RFFit = RFModel.fit(X_train, Y_train)
RFY_Predict = RFModel.predict(X_test)  

RFTrainScore=RFModel.score(X_train,Y_train)
RFTestScore = RFModel.score(X_test,Y_test)



from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=100.0, random_state=0)

SVMscore=cross_val_score(svm,X_train,Y_train,cv=10 )
SVMscore.mean()

svm.fit(X_train,Y_train)
SVMY_Predict=svm.predict(X_test)
SVMTrainScore= svm.score(X_train,Y_train)
SVMTestScore= svm.score(X_test,Y_test)


from sklearn.linear_model import LogisticRegression
Regression=LogisticRegression()
LRscore=cross_val_score(Regression,X_train,Y_train,cv=10 )
LRscore.mean()

Regression.fit(X_train,Y_train)
LRY_Predict=Regression.predict(X_test)
LRTrainScore= Regression.score(X_train,Y_train)
LRTestScore= Regression.score(X_test,Y_test)

#---------- Confusion Matrix---------------------
from sklearn.metrics import confusion_matrix
XGBConfusionMatrix = confusion_matrix(Y_test, XGBY_pred)

RFBConfusionMatrix = confusion_matrix(Y_test, RFY_Predict)

SVMConfusionMatrix = confusion_matrix(Y_test, SVMY_Predict)

LRConfusionMatrix = confusion_matrix(Y_test, LRY_Predict)

#----------------- ROC Curve --------------------
import matplotlib.pyplot as plt

fpr, tpr, threshold=metrics.roc_curve(Y_test,XGBY_pred)
plt.show()

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()





