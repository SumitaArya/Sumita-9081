#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd


# In[63]:


df_reg_user = pd.read_csv("C:\\Users\\Samiksha\\Downloads\\Registered user.csv")


# In[64]:


df_reg_user.head()


# In[167]:


df_reg_user.shape


# In[65]:


df_reg_user.head(30)


# In[66]:


df_reg_user.tail(5)


# In[67]:


df_reg_user.dtypes


# In[68]:


df_reg_user.describe


# In[69]:


df_reg_user.info()


# In[70]:


df_reg_user.isnull().sum().any()


# In[71]:


df_reg_user.isnull().sum()


# In[72]:


df_reg_user.shape


# In[73]:


df_reg_user["FULL ADDRESS"].unique()


# In[74]:


df_red = pd.read_csv("C:\\Users\\Samiksha\\Downloads\\Redeem request.csv")


# In[75]:


df_red


# In[76]:


df_red.info()


# In[77]:


df_red.dtypes


# In[78]:


df_red.describe()


# In[79]:


df_red.isnull().sum()


# In[80]:


df_red.shape


# In[81]:


master_df = df_reg_user.join(df_red)


# In[82]:


master_df


# In[83]:


master_df.info()


# In[84]:


master_df.dtypes


# In[85]:


master_df.describe()


# In[86]:


master_df.isnull().sum().any()


# In[87]:


master_df.nunique()


# In[88]:


master_df["7"].nunique()


# In[89]:


master_df["7"].unique()


# In[90]:


master_df["7"].value_counts()


# In[91]:


df1 = (master_df["7"]== 'SUCCESS').value_counts()


# In[92]:


df1


# In[93]:


df1_new = master_df[master_df["7"]=="SUCCESS"]


# In[94]:


df1_new


# In[95]:


df1_new["7"].value_counts()


# In[96]:


master_df.keys()


# In[97]:


master_df["MOBILE #"].value_counts()


# In[98]:


master_df["MOBILE #"].nunique()


# In[99]:


master_df["11"].value_counts()


# In[100]:


master_df["NAME"].unique()


# In[101]:


master_df["NAME"].value_counts()


# In[102]:


master_df["FULL ADDRESS"].value_counts()


# In[103]:


# So from above we could infer that Maximum Registered user from  Jaipur Rajasthan.


# In[104]:


cols_list = ["NAME" , "MOBILE #", "FULL ADDRESS" ,"ACCOUNT#","7","11"]


# In[105]:


df2=master_df[cols_list]


# In[106]:


df2


# In[107]:


# Below is the list of that user who had more than 500 points .


# In[230]:


master_df.loc[master_df["11"]<500 ,["NAME","MOBILE #","FULL ADDRESS","11" ]].value_counts()


# In[262]:


master_df.loc[master_df["11"]<500 ,["NAME","MOBILE #","FULL ADDRESS","11" ]].value_counts().head(10).plot.pie()


# In[109]:


# Above list of that user who have less than 500 points.


# In[231]:


master_df.loc[master_df["11"]>500 ,["NAME","MOBILE #","FULL ADDRESS" ,"11"]].value_counts() # more than 500 points painters data of bt.


# In[264]:


master_df.loc[master_df["11"]>500 ,["NAME","MOBILE #","FULL ADDRESS" ,"11"]].value_counts().plot.pie()


# In[232]:


master_df.loc[master_df["11"]>1000 ,["NAME","MOBILE #","FULL ADDRESS" ,"11"]].value_counts() # more than 1000 points painters data of bt.


# In[265]:


master_df.loc[master_df["11"]>1000 ,["NAME","MOBILE #","FULL ADDRESS" ,"11"]].value_counts().plot.pie()


# In[ ]:


so from above we could infer that "AKASH(7302039651)" from mundaka delhi under shri shyam paints party. painter is top user for bt points.


# In[111]:


master_df[cols_list].sort_values(by="11" , ascending ='False') # sort data no.of hit wise point means lowest hit to highest hit Painters from june 2019 to 31 july 2020.


# In[223]:


master_df[cols_list].sort_values(by="11" , ascending ='False').head(10) # lowest hit of redeem req user below 10 from june 2019 to 31 july 2020 


# In[113]:


master_df.groupby("ACCOUNT#").count().head(20)


# In[114]:


master_df.groupby("ACCOUNT#")["MOBILE #"].agg("count").tail(20)     # SO WATEVER U SPECIFIED IN THE ROUND BRACES SO WHICH IS GROUPBY IS PUT ON ROUND BRACES along with miles or watelese & aggeregate ()after it.


# In[115]:


# so 99201000015631 this account is register with two mobile number.


# In[116]:


master_df["7"].value_counts().plot.pie()


# In[117]:


master_df["7"].value_counts().plot.bar()


# In[256]:


df_mobile =master_df["MOBILE #"].value_counts().head(20)


# In[258]:


df_mobile.plot.bar()


# In[121]:


master_df["STATE"].value_counts()


# In[122]:


#We could infer from above that maximum registred painters from up. state after that Delhi.


# In[126]:


master_df[master_df['7'] == 1].sort_values('MOBILE #')['FULL ADDRESS']


# In[ ]:


import matplotlib.pyplot as plt


# In[127]:


df2["7"].describe()


# In[140]:


master_df["AADHAR"].value_counts()


# In[142]:


master_df["AADHAR"].describe()


# In[ ]:


We could infer from the above that out of 129 people has same adhar in their detail.


# In[144]:


master_df["ACCOUNT#"].describe()


# In[ ]:


we could infer from the above that 189 bank details are repeated against the numbers


# In[150]:


master_df["ACCOUNT#"].describe()


# In[158]:


master_df[cols_list].sort_values(by="ACCOUNT#").value_counts().tail(50)


# In[168]:


master_df["ACCOUNT#"].drop_duplicates()


# In[175]:


master_df["FIRM TYPE"].value_counts()


# In[ ]:


# From above we can say painters are 7403,contractor are 1307,end cutomer are 57 n so on.


# In[180]:


master_df["BANK NAME"].value_counts()


# In[181]:


master_df["BANK NAME"].describe()


# In[ ]:


# highest bank details from state bank of india.


# In[248]:


master_df.groupby("STATE")["11"].value_counts()


# In[ ]:


# From above we could infer the 250,500,1000 bank trsnffer data state wise.


# In[249]:


master_df.groupby("STATE")["7"].value_counts()


# In[ ]:


So Rajasthan Sate on the top for BT.(from june 2019 to july 2020)
and Uttar Pradesh is on top(3600) for registartion.(from 2019 to till date)
and SBI bank detail given by mostly painters for bt process registartion.
and 189 bank details are repeated against the painters out of 8824.
129 people have the same repeat aadhar number of out of 7921.
painters=7403, contractors= 1307, end customer= 57, or other u can see from 175 line.
AKASH from Mundaka delhi painter has max transffer in one year june 2019 to july 2020 i.e 7500, 2nd is ANKUR SONI from Morena mp 4900 bt,and third is somveer mistri from kirti nagar delhi.

