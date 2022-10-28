#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train_df = pd.read_csv("C:\\Users\\Samiksha\\Downloads\\train.csv")


# In[3]:


train_df.head(30)


# In[4]:


test_df = pd.read_csv("C:\\Users\\Samiksha\\Downloads\\test.csv")


# In[5]:


test_df.head(30)


# In[6]:


print("train info")
train_df.info()   



print("test info")
test_df.info()


# In[7]:


train_df.describe()


# In[8]:


test_df.describe()


# In[9]:


# Shape of train & test data

(train_df.shape) , (test_df.shape)


# In[10]:


train_df.dtypes.value_counts()


# In[11]:


train_df.describe(include = "object")


# In[12]:


train_df.describe(include = "float64")


# In[13]:


train_df.describe(include ="int")


# In[14]:


test_df.dtypes.value_counts()


# In[15]:


test_df.describe(include = "object")


# In[16]:


test_df.describe(include = "int")


# Checking null value in train & test dataset

# In[17]:


train_df.isnull().sum().any()


# In[18]:


test_df.isnull().sum().any()


# In[19]:


train_df.nunique()


# In[20]:


test_df.nunique()


# Checking the train variance in dataset

# In[21]:


train_df.var()


# # finding the columns with variance with zero in train datasets

# In[22]:


train_var = train_df.nunique()
col_to_drop = train_var[train_var ==1].index


# In[23]:


# Cloumns with zero variance 
col_to_drop


# In[24]:


train_df_var = train_df.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293',
                              'X297', 'X330', 'X347' ], axis=1 )


# In[25]:


train_df_var


# # Checking columns with zeri variance in train dataset

# In[26]:


test_var = test_df.nunique()


# In[27]:


cols_to_drop = test_var[test_var ==1].index


# In[28]:


cols_to_drop


# In[29]:


test_df_var = test_df.drop(['X257', 'X258', 'X295', 'X296', 'X369'] , axis =1)


# In[30]:


test_df_var


# In[31]:


(train_df.shape) , (train_df_var.shape)


# In[32]:


(test_df.shape) , (test_df_var.shape)


# # Now Label Encoding

# In[33]:


from sklearn.preprocessing import LabelEncoder


# In[34]:


le = LabelEncoder()


# In[35]:


# Dividing train data into target & features

X_train = train_df_var.drop(["y" , "ID"] , axis =1)
y_train = train_df_var["y"]


# In[36]:


X_train.head(30)


# In[37]:


y_train.head(30)


# In[38]:


# Shape of x & y train
(X_train.shape) , (y_train.shape)


# # Similary dividing test data into train & test dataset

# In[39]:


X_test = test_df_var.drop("ID" , axis =1)
y_test = test_df_var["ID"]


# In[40]:


X_test.head(30)


# In[41]:


y_test.head(30)


# In[42]:


# Checking shape of X & y test data

(X_test.shape) , (y_test.shape)


# In[43]:


X_train.dtypes


# In[44]:


X_train.describe(include = "object")


# In[45]:


X_train.describe(include = "int64")


# In[46]:


y_train.dtypes


# In[47]:


X_test.dtypes


# In[48]:


X_test.describe(include = "object")


# In[49]:


X_test.describe(include = "int64")


# In[50]:


y_test.dtypes


# # Now converting X_train data into int

# In[51]:


X_train['X0'] = le.fit_transform(X_train.X0)
X_train['X1'] = le.fit_transform(X_train.X1)
X_train['X2'] = le.fit_transform(X_train.X2)
X_train['X3'] = le.fit_transform(X_train.X3)
X_train['X4'] = le.fit_transform(X_train.X4)
X_train['X5'] = le.fit_transform(X_train.X5)
X_train['X6'] = le.fit_transform(X_train.X6)
X_train['X8'] = le.fit_transform(X_train.X8)


# In[52]:


X_train.dtypes.value_counts()


# In[53]:


# Similary label encoding for X_test dataset

X_test['X0'] = le.fit_transform(X_test.X0)
X_test['X1'] = le.fit_transform(X_test.X1)
X_test['X2'] = le.fit_transform(X_test.X2)
X_test['X3'] = le.fit_transform(X_test.X3)
X_test['X4'] = le.fit_transform(X_test.X4)
X_test['X5'] = le.fit_transform(X_test.X5)
X_test['X6'] = le.fit_transform(X_test.X6)
X_test['X8'] = le.fit_transform(X_test.X8)


# In[54]:


X_test.dtypes.value_counts()


# # Importing Principal Component Analysis to perform Dimesionality Reduction

# In[55]:


from sklearn.decomposition import PCA


# In[56]:


pca = PCA(n_components=10, random_state =10)


# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


X_train.shape,y_train.shape,X_train.shape,X_test.shape


# In[59]:


(train_df.shape) , (test_df.shape) # Dataframe before using pca 


# In[61]:


# Using fit.transform function in X train Xtest.
X_train_pca= pca.fit_transform(X_train)
X_test_pca= pca.fit_transform(X_test)


# In[62]:


X_train.shape , X_test.shape    # after using pca trasnform function


# # Using XGBoost method for tran_test_split function for training & testing prediction

# In[115]:


import xgboost as xgb


# In[116]:


# For efficient use of the XGBoost model we are planning to convert our dataset to the Dmatrix format

#Dmatrix is the data structure thast is unique to the XGBoost algorithm


D_train = xgb.DMatrix(X_train , label = y_train)
D_test = xgb.DMatrix(X_test)


# In[120]:


param = {"eta": 0.02,"objective": "reg:squarederror"}


# In[121]:


xgb_model =xgb.train(param,D_train,1000)


# # predicting the target value of xgboost model for train data

# In[132]:


y_pred_test_xgb= xgb_model.predict(D_test)


# In[133]:


y_pred_test_xgb


# In[134]:


y_pred_test_xgb.shape


# In[135]:


y_test.shape


# # RMSE of test dataset

# In[136]:


import numpy as np
from sklearn.metrics import r2_score,mean_squared_error


# In[139]:


RMSE = print(np.sqrt(mean_squared_error(y_test,y_pred_test_xgb)))


# # So our RMSE value is below 10.

# # Now check RMSE for y_train

# In[140]:


y_pred_train_xgb= xgb_model.predict(D_train)


# In[141]:


RMSE = print(np.sqrt(mean_squared_error(y_train,y_pred_train_xgb)))


# So our RMSE value of training dataset is 4.

# # Now calculating r2 score for test dataset

# In[183]:


from sklearn.metrics import r2_score,mean_squared_error


# In[184]:


R2_score = r2_score(y_pred_test_xgb,y_test)


# In[185]:


R2_score


# # Now calculating r2 score for train dataset

# In[186]:


R2_score = r2_score(y_pred_train_xgb,y_train)


# In[187]:


R2_score

