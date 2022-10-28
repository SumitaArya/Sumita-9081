#!/usr/bin/env python
# coding: utf-8

# # Capstone Project: Healthcare
#     
#  # Problem Statement:
# 
# NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases) research creates knowledge about and treatments for the most chronic, costly, and consequential diseases.
# The dataset used in this project is originally from NIDDK. The objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
# Build a model to accurately predict whether the patients in the dataset have diabetes or not.
# Dataset Description: The datasets consists of several medical predictor variables and one target variable (Outcome). Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and more.
# 
# Variables - Description
# 
# Pregnancies - Number of times pregnant
# Glucose - Plasma glucose concentration in an oral glucose tolerance test
# BloodPressure - Diastolic blood pressure (mm Hg)
# SkinThickness - Triceps skinfold thickness (mm)
# Insulin - Two hour serum insulin
# BMI - Body Mass Index
# DiabetesPedigreeFunction - Diabetes pedigree function
# Age - Age in years
# Outcome - Class variable (either 0 or 1). 268 of 768 values are 1, and the others are 0
# Week 1:
# Data Exploration:
# 
# Perform descriptive analysis. Understand the variables and their corresponding values. On the columns below, a value of zero does not make sense and thus indicates missing value:
# 
#  • Glucose
#  • BloodPressure
#  • SkinThickness
#  • Insulin
#  • BMI
# Visually explore these variables using histograms. Treat the missing values accordingly.
# There are integer and float data type variables in this dataset. Create a count (frequency) plot describing the data types and the count of variables.
# Week 2:
# Data Exploration:
# 
# Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action.
# Create scatter charts between the pair of variables to understand the relationships. Describe your findings.
# Perform correlation analysis. Visually explore it using a heat map.
# Week 3:
# Data Modeling:
# 
# Devise strategies for model building. It is important to decide the right validation framework. Express your thought process.
# Apply an appropriate classification algorithm to build a model. Compare various models with the results from KNN algorithm.
# Week 4:
# Data Modeling:
# 
# Create a classification report by analyzing sensitivity, specificity, AUC (ROC curve), etc. Please be descriptive to explain what values of these parameter you have used.
# Data Reporting:
# 
# Create a dashboard in tableau by choosing appropriate chart types and metrics useful for the business. The dashboard must entail the following:
# 
#  a. Pie chart to describe the diabetic or non-diabetic population
#  b. Scatter charts between relevant variables to analyze the relationships
#  c. Histogram or frequency charts to analyze the distribution of the data
#  d. Heatmap of correlation analysis among the relevant variables
#  e. Create bins of these age values: 20-25, 25-30, 30-35, etc. Analyze different variables for these age brackets using a bubble chart.
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


health_df = pd.read_csv("C:\\Users\\Samiksha\\Downloads\\Project_2\\Project 2\\Healthcare - Diabetes\\health care diabetes.csv")


# In[3]:


health_df


# So this dataset has 768 rows & 9 coulmns.

# In[4]:


health_df.info()


# In[5]:


health_df.describe()


# we could infer the above statastic values of the health diabetic dataset.

# In[6]:


health_df.columns


# In[7]:


import numpy as np


# In[8]:


for i in range(0, len(np.array_split(health_df.dtypes, 10))):
    print((np.array_split(health_df.dtypes, 10)[i]))
    print()


# In[9]:


health_df[health_df.columns[:10]].head(20)


# In[10]:


for i in range(0, len(health_df.columns), 40):
    print(health_df[health_df.columns[i:i+40]].head())
    print()


# In[11]:


cat_columns = ["Pregnancies", "Glucose" ,"BloodPressure" ,"SkinThickness" ,"Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]


# In[12]:


health_df[cat_columns].dtypes


# In[13]:


for col in cat_columns:
    print(col)
    print(health_df[col].nunique())
    print(health_df[col].unique())
    print()


# In[14]:


health_df['Glucose'].value_counts().head(10)


# In[15]:


health_df["Glucose"].value_counts()[0]


# In[16]:


plt.hist(health_df["Glucose"])


# In[17]:


health_df["Glucose"].mean()


# In[18]:


health_df["Glucose"] = health_df["Glucose"].replace(0,120.89453125)


# In[19]:


health_df["Glucose"]


# In[20]:


plt.hist(health_df["Glucose"])


# I am treating missing values which is basically 0 by mean of Glucose level, This is because we can see from histogram most of observation have Glucose level between 100 and 120.
# 
# 

# In[21]:


health_df["BloodPressure"].value_counts().head(10)


# In[22]:


plt.hist(health_df["BloodPressure"])


# In[23]:


health_df["BloodPressure"].mean()


# In[24]:


health_df["BloodPressure"]= health_df["BloodPressure"].replace(0,69.10546875)


# In[25]:


health_df["BloodPressure"]


# In[26]:


plt.hist(health_df["BloodPressure"])


# I am treating missing values which is basically 0 by mean of BloodPressure level. This is because we can see from histogram most of observation have BP level between 70 and 80.
# 
# 

# In[27]:


health_df["SkinThickness"].value_counts().head(10)


# In[28]:


plt.hist(health_df["SkinThickness"])


# In[29]:


health_df["SkinThickness"].mean()


# In[30]:


health_df["SkinThickness"]= health_df["SkinThickness"].replace(0,20.536458333333332)


# In[31]:


health_df["SkinThickness"]


# In[32]:


plt.hist(health_df["SkinThickness"])


# I am treating missing values which is basically 0 by mean of SkinThickness. This is because we can see from histogram most of observation have SkinThickness between 20 and 30.
# 
# 

# From above histograms, it is clear that Insulin has highly skewed data distribution so we replace zero(null value) by median.
# m

# In[33]:


health_df["Insulin"].value_counts()


# In[34]:


health_df["Insulin"].value_counts()[0]


# In[35]:


plt.hist(health_df["Insulin"])


# In[36]:


health_df["Insulin"].median()


# In[37]:


health_df["Insulin"]= health_df["Insulin"].replace(0,30.5)


# In[38]:


health_df["Insulin"]


# In[39]:


plt.hist(health_df["Insulin"])


# In[40]:


health_df["BMI"].value_counts()


# In[41]:


plt.hist(health_df["BMI"])


# In[42]:


health_df["BMI"].value_counts()[0]


# In[43]:


health_df["BMI"]= health_df["BMI"].replace(0,31.992578)


# In[44]:


health_df["BMI"]


# In[45]:


plt.hist(health_df["BMI"])


# In[46]:


health_df.describe()


# so from above we could infer the statistic values of the health dataset.`

# In[47]:


health_df.describe().T


# (3) Create a count (frequency) plot describing the data types and the count of variables:
# 
# 

# In[48]:


health_df.dtypes.value_counts().plot(kind='bar')


#  So, There are integer and float data type variables in this dataset,We have created a count (frequency) plot describing the data types and the count of variables. 
# 
# 

#  # Project Task: Week 2
# 

# Data Exploration:
# 

# 1. Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action.
# 

# From above histograms, it is clear that Insulin has highly skewed data distribution and remaining 4 variables have relatively balanced data distribution therefore we will treat missing values in these 5 variables as below:-
# 
# Glucose - replace missing values with mean of values.
# 
# BloodPressure - replace missing values with mean of values.
# 
# SkinThickness - replace missing values with mean of values.
# 
# Insulin - replace missing values with median of values.
# 
# BMI - replace missing values with mean of values.
# 

# 
# 
# 
# 
# # Week 2:
#     
# Data Exploration:
#     
# (1) Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action:
# 
# 

# In[49]:


health_df["Outcome"].value_counts()


# In[50]:


health_df["Outcome"].value_counts().plot(kind ="barh")


# In[51]:


sns.displot(health_df["Outcome"])
plt.show()


# In[52]:


health_df["Outcome"].value_counts().plot(kind ="bar")


# 
# So,classes in Outcome is little skewed so we will generate new samples using SMOTE (Synthetic Minority Oversampling Technique) for the class '1' which is under-represented in our data. We will use SMOTE out of many other techniques available since:
# 
# It has generates new samples by interpolation.
# 
# It doesn't duplicate data.
# 

# So Outcome is our target variable in above dataset

# In[53]:


health_df_X = health_df.drop("Outcome" ,axis=1)
health_df_y = health_df["Outcome"]


# In[54]:


print(health_df_X.shape)


# In[55]:


print(health_df_y.shape)


# In[56]:


pip install imbalanced-learn


# In[57]:


from imblearn.over_sampling import SMOTE


# In[58]:


health_df_X_resampled, health_df_y_resampled = SMOTE(random_state=108).fit_resample(health_df_X,health_df_y)


# In[59]:


print(health_df_X_resampled.shape, health_df_y_resampled.shape)


#  # Now we create a count plot for Outcome after done SMOTE

# In[60]:


health_df_y_resampled.value_counts()


# In[61]:


health_df_y_resampled.value_counts().plot(kind="bar")


#  Now we could infer from above that the data is balanced.

# 
# 2. Create scatter charts between the pair of variables to understand the relationships. Describe your findings.
# 
# 

# In[62]:


health_df_resampled = pd.concat([health_df_X_resampled,health_df_y_resampled],axis=1)


# In[63]:


health_df_resampled


# In[64]:


sns.set(rc={'figure.figsize':(8,8)})
sns.scatterplot(x="Pregnancies", y="Glucose", data=health_df_resampled, hue="Outcome");


# In[65]:


sns.set(rc={"figure.figsize":(8,8)})
sns.scatterplot(x= "Pregnancies", y= "SkinThickness", data = health_df_resampled,hue ="Outcome");


# Now simillary we create all the scatterplots for all varaibles to find out their relationship.

# In[66]:


fig, axes = plt.subplots (8,8, figsize=(22, 17))

fig.suptitle('Scatter Plot for Features in Training Data')

for i, col_y in enumerate(health_df_X_resampled.columns):
    for j, col_x in enumerate(health_df_X_resampled.columns):             
        sns.scatterplot(ax=axes[i, j], x=col_x, y=col_y, data=health_df_resampled, hue="Outcome", legend = False)

plt.tight_layout()


# 
# 
# Some observation we have noticed from above sactter charts of features like mention below:
#     
#     
#     Glucose alone is impressively good to distinguish between the Outcome classes.
#     
#     Age alone is also able to distinguish between classes to some extent.
# 
#     It seems none of pairs in the dataset is able to clealry distinguish between the Outcome classes.
# 
#     We need to use combination of features to build model for prediction of classes in Outcome.
# 

# # (3) Perform correlation analysis. Visually explore it using a heat map:
# 
# 

# In[67]:


corr = health_df_X_resampled.corr()


# In[68]:


corr


# Now we create correlation map for above data

# In[69]:


plt.figure(figsize=(15,10))
sns.heatmap(health_df_X_resampled.corr(), cmap='viridis', annot=True);


# It appears from correlation matrix and heatmap that there exists significant correlation between some pairs such as -
# 
# Age-Pregnancies
# 
# BMI-SkinThickness
# 

# 
# 
# 
#  # Week 3:
# Data Modeling:
#     
# (1) Devise strategies for model building. It is important to decide the right validation framework. Express your thought process:
# 
# 

# 
# 
# Since this is a classification problem, we will be building all popular classification models for our training data and then compare performance of each model on test data to accurately predict target variable (Outcome):
# 
# 

# Performing train & test on input on without cross validation.

# In[70]:


from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(health_df_X_resampled, health_df_y_resampled, test_size=0.15, random_state =10)


# In[72]:


print(X_train.shape, X_test.shape)


# 
# 
# 2. Apply an appropriate classification algorithm to build a model. Compare various models with the results from KNN algorithm.
# 

# In[73]:


from sklearn.metrics import accuracy_score, average_precision_score, f1_score, confusion_matrix, classification_report, auc, roc_curve, roc_auc_score, precision_recall_curve


# In[74]:


models = []
model_accuracy = []
model_f1 = []
model_auc = []


# Now we start with Logistic Regression

# In[75]:


from sklearn.linear_model import LogisticRegression


# In[76]:


lr_model = LogisticRegression(solver="newton-cg")


# In[77]:


lr_model.fit(X_train,y_train)


# In[78]:


y_test_pred = lr_model.predict(X_test)


# In[79]:


y_test_pred


# In[80]:


print("The accuracy score of the model on the test data is : ")
print(accuracy_score(y_test , y_test_pred)) #the first parameter is the actual values & the second parameter always to the predictive values.


# Simillarly finding out the train dataset accuracy score as well

# In[81]:


y_train_pred = lr_model.predict(X_train)


# In[82]:


y_train_pred


# In[83]:


print("The accuracy score of the model on the train data is : ")
print(accuracy_score(y_train , y_train_pred)) 


# Performance evaluation and optimizing parameters using GridSearchCV: Logistic regression does not really have any critical hyperparameters to tune. However we will try to optimize one of its parameters 'C' with the help of GridSearchCV. So we have set this parameter as a list of values form which GridSearchCV will select the best value of parameter.
# 
# 

# In[84]:


from sklearn.model_selection import GridSearchCV, cross_val_score


# In[85]:


parameters = {'C':np.logspace(-5, 5, 50)}


# In[86]:


gs_lr = GridSearchCV(lr_model, param_grid = parameters, cv=5, verbose=0)
gs_lr.fit(health_df_X_resampled, health_df_y_resampled)


# In[87]:


gs_lr.best_params_


# In[88]:


gs_lr.best_score_


# In[89]:


lr_model2 = LogisticRegression(C=13.257113655901108,solver="newton-cg")


# In[90]:


lr_model2.fit(X_train,y_train)


# In[91]:


y_test_pred1 = lr_model2.predict(X_test)


# In[92]:


y_test_pred1


# In[93]:


print("The accuracy score of the model on the test data is : ")
print(accuracy_score(y_test , y_test_pred1)) 


# In[94]:


y_train_pred1 = lr_model2.predict(X_train)


# In[95]:


y_train_pred1


# In[96]:


print("The accuracy score of the model on the train data is : ")
print(accuracy_score(y_train , y_train_pred1)) 


# so now test accuracy is 80 % & train accuracy is 74 % approx .

# 
# 
# # So now we preparing ROC Curve (Receiver Operating Characteristics Curve)
# 

# In[97]:


probs = lr_model2.predict_proba(X_test)                
probs = probs[:, 1]                              

auc_lr = roc_auc_score(y_test, probs)           
print('AUC: %.3f' %auc_lr)
fpr, tpr, thresholds = roc_curve(y_test, probs)  
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')              
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC (Receiver Operating Characteristics) Curve");


# Predict probabilities
# 
# Keep probabilities for the positive outcome only
# 
# Calculate AUC
# 
# Calculate roc curve
#     
# Plot no skill     
# 
# Plot the roc curve for the model

# # Now we find out the classification report of above model

# In[98]:


print(classification_report(y_test , y_test_pred1))


# support is nothing but Total no. of records that were taken to identify precision ,recall , f-1 score, i.e,68 records that belongs to the class-0,& 82 records belongs to the class-1. 

# In[99]:


# Precision Recall Curve 

pred_y_test = lr_model2.predict(X_test)                                     
precision, recall, thresholds = precision_recall_curve(y_test, probs) 
f1 = f1_score(y_test, y_test_pred1)                                    
auc_lr_pr = auc(recall, precision)                                    
ap = average_precision_score(y_test, probs)                           
print('f1=%.3f auc_pr=%.3f ap=%.3f' % (f1, auc_lr_pr, ap))
plt.plot([0, 3], [0.6, 0.6], linestyle='--')                          
plt.plot(recall, precision, marker='.')                               
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve");


# Predict class values
#     
# Calculate precision-recall curve
# 
# Calculate F1 score
#     
# Calculate precision-recall AUC
# 
# Calculate average precision score
# 
# Plot no skill
# 
# Plot the precision-recall curve for the model

# In[100]:


models.append('LR')
model_accuracy.append(accuracy_score(y_test, y_test_pred1))
model_f1.append(f1)
model_auc.append(auc_lr)


# # Now we start with Random forest algorithm

# In[101]:


from sklearn.ensemble import RandomForestClassifier


# In[102]:


rfc_model = RandomForestClassifier()


# In[103]:


rfc_model.fit(X_train,y_train)


# In[104]:


rfc_model_y_test_pred = rfc_model.predict(X_test)


# In[105]:


rfc_model_y_test_pred


# In[106]:


print("The prediction accuracy of RFC test model is : ")
print(accuracy_score(y_test ,rfc_model_y_test_pred))


# So Random forest test data accuracy is 85 % , now we checking for train dataas well.

# In[107]:


rfc_model_y_train_pred = rfc_model.predict(X_train)


# In[108]:


rfc_model_y_train_pred


# In[109]:


print("The prediction accuracy of RFC train model is : ")
print(accuracy_score(y_train ,rfc_model_y_train_pred))


# # Random Forest 100% accuracy over train data always
# 

#  # Performance evaluation and optimizing parameters using GridSearchCV:
# 
# 

# In[110]:


parameters = {
    'n_estimators': [50,100,150],
    'max_depth': [None,1,3,5,7],
    'min_samples_leaf': [1,3,5]
}


# In[111]:


gs_dt = GridSearchCV(estimator=rfc_model, param_grid=parameters, cv=5, verbose=0)
gs_dt.fit(health_df_X_resampled, health_df_y_resampled)


# In[112]:


gs_dt.best_params_


# In[113]:


gs_dt.best_score_


# In[114]:


rfc_model.feature_importances_


# Now we are plotting here chart to show features importances in our above model

# In[115]:


plt.figure(figsize=(10,5))
sns.barplot(y=X_train.columns, x=rfc_model.feature_importances_);
plt.title("Feature Importance in Model");


# In[116]:


rfc_model2 = RandomForestClassifier(max_depth=None, min_samples_leaf=1, n_estimators=100)


# In[117]:


rfc_model2


# In[118]:


rfc_model2.fit(X_train,y_train)


# In[119]:


rfc_model2_y_test_pred2 = rfc_model2.predict(X_test)


# In[120]:


rfc_model2_y_test_pred2


# In[121]:


print("The prediction accuracy of RFC of test model is : ")
print(accuracy_score(y_test ,rfc_model2_y_test_pred2))


# So now test accuracy is 85 % now we chk train accuracy.

# In[122]:


rfc_model2_y_train_pred2 = rfc_model2.predict(X_train)


# In[123]:


rfc_model2_y_train_pred2


# In[124]:


print("The prediction accuracy of RFC of train model is : ")
print(accuracy_score(y_train ,rfc_model2_y_train_pred2))


# So rfc_model2 train model accuracy is 100 % as its is.

# In[125]:


# Preparing ROC Curve (Receiver Operating Characteristics Curve)

probs = rfc_model2.predict_proba(X_test)                
probs = probs[:, 1]                              

auc_rf = roc_auc_score(y_test, probs)            
print('AUC: %.3f' %auc_rf)
fpr, tpr, thresholds = roc_curve(y_test, probs)  
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC (Receiver Operating Characteristics) Curve");


# Predict probabilities
# 
# Keep probabilities for the positive outcome only
# 
# Calculate AUC
# 
# Calculate roc curve
# 
# Plot no skill
# 
# Plot the roc curve for the model
# 

# In[126]:


# Precision Recall Curve 

pred_y_test = rfc_model2.predict(X_test)                                     
precision, recall, thresholds = precision_recall_curve(y_test, probs) 
f1 = f1_score(y_test, pred_y_test)                                    
auc_rf_pr = auc(recall, precision)                                    
ap = average_precision_score(y_test, probs)                           
print('f1=%.3f auc_pr=%.3f ap=%.3f' % (f1, auc_rf_pr, ap))
plt.plot([0, 1], [0.8, 0.8], linestyle='--')                          
plt.plot(recall, precision, marker='.')                             
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve");


# Predict class values
# 
# Calculate precision-recall curve
# 
# Calculate F1 score
# 
# Calculate precision-recall AUC
# 
# Calculate average precision score
# 
# Plot no skill
# 
# Plot the precision-recall curve for the model
# 

# In[127]:


models.append('RF')
model_accuracy.append(accuracy_score(y_test, pred_y_test))
model_f1.append(f1)
model_auc.append(auc_rf)


# # Now we test with Decision Tree algorithm

# In[128]:


from sklearn.tree import DecisionTreeClassifier


# In[129]:


dt_model  = DecisionTreeClassifier(random_state=0)     


# In[130]:


dt_model.fit(X_train,y_train)
  


# In[131]:


dt_model_y_test_pred = dt_model.predict(X_test)


# In[132]:


dt_model_y_test_pred


# In[133]:


print("The prediction accuracy score of Decision tree model on testing data is :")
print(accuracy_score(y_test , dt_model_y_test_pred))


# Now checking for training dataset also

# In[134]:


dt_model_y_train_pred = dt_model.predict(X_train)


# In[135]:


dt_model_y_train_pred


# In[136]:


print("The prediction accuracy score of Decision tree model on training data is :")
print(accuracy_score(y_train , dt_model_y_train_pred))


# So accuracy of Decision Tree model train dataset is 100 % and test dataset is approx 76 %.

# Performance evaluation and optimizing parameters using GridSearchCV:
# 
# 

# In[137]:


parameters = {
    'max_depth':[1,2,3,4,5,None]
}


# In[138]:


gs_dt = GridSearchCV(dt_model, param_grid = parameters, cv=5, verbose=0)
gs_dt.fit(health_df_X_resampled, health_df_y_resampled)


# In[139]:


gs_dt.best_params_


# In[140]:


gs_dt.best_score_


# In[141]:


dt_model.feature_importances_


# shows in chart form these values of columns

# In[142]:


plt.figure(figsize=(12,5))
sns.barplot(y=X_train.columns, x=dt_model.feature_importances_)
plt.title("Feature Importance in Model");


# In[143]:


dt_model1 = DecisionTreeClassifier(max_depth=4)


# In[144]:


dt_model1 =dt_model1.fit(X_train,y_train)


# In[145]:


dt_model1_y_test_pred1 = dt_model1.predict(X_test)


# In[146]:


dt_model1_y_test_pred1


# In[147]:


print("The prediction accuracy score of Decision tree model on testing data is :")
print(accuracy_score(y_test , dt_model1_y_test_pred1))


# Now checking for traning dataset also

# In[148]:


dt_model1_y_train_pred1 = dt_model1.predict(X_train)


# In[149]:


dt_model1_y_train_pred1


# In[150]:


print("The prediction accuracy score of Decision tree model on traning data is :")
print(accuracy_score(y_train , dt_model1_y_train_pred1))


# So Decision Tree testing & traning is almost same 79 % approx.

# 
# 
# Preparing ROC Curve (Receiver Operating Characteristics Curve)
# 

# In[151]:


probs = dt_model1.predict_proba(X_test)                
probs = probs[:, 1]                              
auc_dt = roc_auc_score(y_test, probs)            
print('AUC: %.3f' %auc_dt)
fpr, tpr, thresholds = roc_curve(y_test, probs)  
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC (Receiver Operating Characteristics) Curve");


# Predict probabilities
# 
# Keep probabilities for the positive outcome only
# 
# Calculate AUC
# 
# Calculate roc curve
# 
# Plot no skill
# 
# Plot the roc curve for the model
# 

# 
# 
# # Precision Recall Curve 
# 

# In[152]:


pred_y_test = dt_model1.predict(X_test)                                     
precision, recall, thresholds = precision_recall_curve(y_test, probs) 
f1 = f1_score(y_test, pred_y_test)                                   
auc_dt_pr = auc(recall, precision)                                    
ap = average_precision_score(y_test, probs)                           
print('f1=%.3f auc_pr=%.3f ap=%.3f' % (f1, auc_dt_pr, ap))
plt.plot([0, 1], [0.6, 0.6], linestyle='--')                          
plt.plot(recall, precision, marker='.')                               
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve");


# Predict class values
# 
# Calculate precision-recall curve
# 
# Calculate F1 score
# 
# Calculate precision-recall AUC
# 
# Calculate average precision score
# 
# Plot no skill
# 
# Plot the precision-recall curve for the model
# 

# In[153]:


models.append('DT')
model_accuracy.append(accuracy_score(y_test, pred_y_test))
model_f1.append(f1)
model_auc.append(auc_dt)


# 
# 
# #  K-Nearest Neighbour (KNN) Classification:
# 

# In[154]:


from sklearn.neighbors import KNeighborsClassifier


# In[155]:


knn_model = KNeighborsClassifier(n_neighbors=3)


# In[156]:


knn_model.fit(X_train,y_train)


# In[157]:


knn_model_y_test_pred = knn_model.predict(X_test)


# In[158]:


knn_model_y_test_pred


# In[159]:


print("The accuracy score of knn_model on test data is :")
print(accuracy_score(y_test , knn_model_y_test_pred))


# Now chekcing for train dataset also

# In[160]:


knn_model_y_train_pred = knn_model.predict(X_train)


# In[161]:


knn_model_y_train_pred


# In[162]:


print("The accuracy score of the model on train data is :")
print(accuracy_score(y_train , knn_model_y_train_pred))


# So accuracy for train dataset is 89 % and for test data is 78 % .

# Now Performance evaluation and optimizing parameters using GridSearchCV:
# 

# In[163]:


knn_neighbors = [i for i in range(1,25)]
parameters = {
    'n_neighbors': knn_neighbors
}


# In[164]:


gs_knn = GridSearchCV(estimator=knn_model, param_grid=parameters, cv=5, verbose=0)
gs_knn.fit(health_df_X_resampled, health_df_y_resampled)


# In[165]:


gs_knn.best_params_


# In[166]:


gs_knn.best_score_


# In[167]:


# gs_knn.cv_results_
gs_knn.cv_results_['mean_test_score']


# Now plotting chart for testing accuracy score

# In[168]:


plt.figure(figsize=(17,4))
sns.barplot(x=knn_neighbors, y=gs_knn.cv_results_['mean_test_score'])
plt.xlabel("N_Neighbors")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs. N_Neighbors");


# In[169]:


knn_model1 = KNeighborsClassifier(n_neighbors=3)


# In[170]:


knn_model1.fit(X_train,y_train)


# In[171]:


knn_model1_y_test_pred1 = knn_model1.predict(X_test)


# In[172]:


knn_model1_y_test_pred1


# In[173]:


print("The accuracy score of the model on test data is :")
print(accuracy_score(y_test , knn_model1_y_test_pred1))


# In[174]:


knn_model1_y_train_pred1 = knn_model1.predict(X_train)


# In[175]:


knn_model1_y_train_pred1


# In[176]:


print("The accuracy score of the model on train data is :")
print(accuracy_score(y_train , knn_model1_y_train_pred1))


# So accuracy_score for testing dataset for knn_model1 is 78 % and training is 89 % which is same as previous knn_model.

# 
# 
#  # Now Preparing ROC Curve (Receiver Operating Characteristics Curve)
# 

# In[177]:


probs = knn_model1.predict_proba(X_test)              
probs = probs[:, 1]                             
auc_knn = roc_auc_score(y_test, probs)         
print('AUC: %.3f' %auc_knn)
fpr, tpr, thresholds = roc_curve(y_test, probs)  
plt.plot([0, 1], [0, 1], linestyle='--')       
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC (Receiver Operating Characteristics) Curve");


# Predict probabilities
# 
# Keep probabilities for the positive outcome only
# 
# Calculate AUC
# 
# Calculate roc curve
# 
# Plot no skill
# 
# Plot the roc curve for the model
# 

# 
# 
#  # Now Precision Recall Curve 
# 

# In[178]:


pred_y_test = knn_model1.predict(X_test)                                  
precision, recall, thresholds = precision_recall_curve(y_test, probs) 
f1 = f1_score(y_test, pred_y_test)                                    
auc_knn_pr = auc(recall, precision)                                    
ap = average_precision_score(y_test, probs)                           
print('f1=%.3f auc_pr=%.3f ap=%.3f' % (f1, auc_knn_pr, ap))
plt.plot([0, 1], [0.4, 0.4], linestyle='--')                        
plt.plot(recall, precision, marker='.')                               
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve");


# Predict class values
# 
# Calculate precision-recall curve
# 
# Calculate F1 score
# 
# Calculate precision-recall AUC
# 
# Calculate average precision score
# 
# Plot no skill
# 
# Plot the precision-recall curve for the model
# 

# In[179]:


models.append('KNN')
model_accuracy.append(accuracy_score(y_test, pred_y_test))
model_f1.append(f1)
model_auc.append(auc_knn)


# # Now Checking with  Naive Bayes Algorithm:
# 

# In[180]:


from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


# In[181]:


gnb_model = GaussianNB()


# In[182]:


gnb_model.fit(X_train, y_train)


# In[183]:


gnb_model_y_test_pred = gnb_model.predict(X_test)


# In[184]:


gnb_model_y_test_pred


# In[185]:


print("The prediction accuracy score of Naive Bayes Classifier model on test data is :")
print(accuracy_score(y_test , gnb_model_y_test_pred))


# In[186]:


gnb_model_y_train_pred = gnb_model.predict(X_train)


# In[187]:


gnb_model_y_train_pred


# In[188]:


print("The prediction accuracy score of Naive Bayes Classifier model on train data is :")
print(accuracy_score(y_train , gnb_model_y_train_pred))


# So by using Naive Bayes Classifier Model the test accuracy core is arround 80 % and train is around 73 %.

# Naive Bayes has almost no hyperparameters to tune.
# 
# 

# # Now Preparing ROC Curve (Receiver Operating Characteristics Curve)
# 

# In[189]:


probs = gnb_model.predict_proba(X_test)                
probs = probs[:, 1]                         
auc_gnb = roc_auc_score(y_test, probs)          
print('AUC: %.3f' %auc_gnb)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC (Receiver Operating Characteristics) Curve");


# Predict probabilities
# 
# Keep probabilities for the positive outcome only
# 
# Calculate AUC
# 
# Calculate roc curve
# 
# Plot no skill
# 
# Plot the roc curve for the model
# 

# # Now for Precision Recall Curve 
# 
# 

# In[190]:


pred_y_test = gnb_model.predict(X_test)                                     
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1 = f1_score(y_test, pred_y_test)                                   
auc_gnb_pr = auc(recall, precision)                                   
ap = average_precision_score(y_test, probs)                          
print('f1=%.3f auc_pr=%.3f ap=%.3f' % (f1, auc_gnb_pr, ap))
plt.plot([0, 1], [0.6, 0.6], linestyle='--')                         
plt.plot(recall, precision, marker='.')                               
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve");


# Predict class values
# 
# Calculate precision-recall curve
# 
# Calculate F1 score
# 
# Calculate precision-recall AUC
# 
# Calculate average precision score
# 
# Plot no skill
# 
# Plot the precision-recall curve for the model
# 

# In[191]:


models.append('GNB')
model_accuracy.append(accuracy_score(y_test, pred_y_test))
model_f1.append(f1)
model_auc.append(auc_gnb)


# In[192]:


model_summary = pd.DataFrame(zip(models,model_accuracy,model_f1,model_auc), columns = ['model','accuracy','f1_score','auc'])
model_summary = model_summary.set_index('model')


# In[193]:


model_summary.plot(figsize=(21,8))
plt.title(" The Comparison of Different Classification Algorithms");


# In[194]:


model_summary


# So we can see from above model_summary is that the  Random Forest Classifier is best among all, you might be wondering auc score is lesser by 1 than others also i am considering it to be best because balance of classes between Precision and Recall is far better than other Models. So we can consider a loss in AUC by 1 anf test accuracy score is around 84 % approx and auc value is around 92 %.

# # Among all models, RandomForest has given best accuracy and f1_score. Therefore we will build final model using RandomForest.
# 
# 

# # Week 4:
#     
# Data Modeling:
#     
# (1) Create a classification report by analyzing sensitivity, specificity, AUC (ROC curve), etc. Please be descriptive to explain what values of these parameter you have used:
# 
# 

# In[195]:


model_final = rfc_model2


# In[196]:


print(classification_report(y_test , model_final.predict(X_test)))


# So we can see from above the classification report of our final model which is clearly seen above.

# In[197]:


confusion = confusion_matrix(y_test, model_final.predict(X_test))
print("Confusion Matrix: \n", confusion)


# In[198]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Sensitivity = TP/(TP+FN)                     
Specificity = TN/(TN+FP)


# Acuuracy,Precision,Sensitivity,Specificty find out from above their values print after that.

# In[199]:


print("Accuracy: %.3f"%Accuracy)
print("Precision: %.3f"%Precision)
print("Sensitivity: %.3f"%Sensitivity)
print("Specificity: %.3f"%Specificity)
print("AUC: %.3f"%auc_rf)


# So we could infer from abov that the Accuracy is around 83 % ,Precision is around 83 %, Sensistivity is around 86 % , Specificty is around 79 % and AUC va;ue is aorund 91 %.

# Ideally we want to maximize both Sensitivity & Specificity. But this is not possible always. There is always a trade-off. So Sometimes we want to be 100% sure on Predicted negatives values, sometimes we want to be 100% sure on Predicted positives. Sometimes we simply don’t want to compromise on sensitivity sometimes we don’t want to compromise on specificity.
# 
# 

# The threshold is set based on business problem. There are some cases where Sensitivity is important and need to be near to 1. There are business cases where Specificity is important and need to be near to 1. 
# 
# 

# # Data Reporting:
# 

# # 2. Create a dashboard in tableau by choosing appropriate chart types and metrics useful for the business. The dashboard must entail the following:
# 
#     a. Pie chart to describe the diabetic or non-diabetic population
#     b. Scatter charts between relevant variables to analyze the relationships
#     c. Histogram or frequency charts to analyze the distribution of the data
#     d. Heatmap of correlation analysis among the relevant variables
#     e. Create bins of these age values: 20-25, 25-30, 30-35, etc. Analyze different variables for these age brackets using a bubble chart.
# 

# # So Exporting Dataset to csv file 

# In[200]:


health_df.head()


# In[202]:


health_df.isnull().sum().any()


# In[204]:


health_df.to_csv("C:\\Users\\Samiksha\\Downloads\\Project_2\\Project 2\\Healthcare - Diabetes\\health care diabetes_new.csv")


# Now importing the above dataset into Tableu for making Dashboard.

# In[ ]:




