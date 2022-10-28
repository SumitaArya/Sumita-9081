#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_df = pd.read_csv("C:\\Users\\Samiksha\\Downloads\\Project_1\\Project 1\\train.csv")


# In[3]:


train_df.head(30)


# 2.figure out the primary key and looking for the indexing

#  # so we can UID is the primary key

# 3. Gauge the fill rate of the variables and devise plans for missing value treatment ,so explain explicitly the reason for the treatment chosen for each variable.
# 

# In[4]:


train_df.columns


# In[5]:


train_df.dtypes


# In[6]:


train_df.info()


# In[7]:


train_df.describe()


# In[8]:


train_df.columns[:10]


# In[9]:


import numpy as np


# In[10]:


for i in range(0, len(np.array_split(train_df.dtypes, 10))):
    print((np.array_split(train_df.dtypes, 10)[i]))
    print()


# In[11]:


train_df[train_df.columns[:30]].head(30)


# In[12]:


for i in range(0, len(train_df.columns), 30):
    print(train_df[train_df.columns[i:i+30]].head())
    print()
 


# In[13]:


category_columns = ['UID', 'COUNTYID', 'STATEID', 'state', 'state_ab', 'city', 'place', 'type', 'primary', 'zip_code', 'area_code']


# In[14]:


train_df[category_columns].dtypes


# In[15]:


for col in category_columns:
    print(col)
    print(train_df[col].nunique())
    print(train_df[col].unique())
    print()


# Now checking the null values in train dataset

# In[16]:


train_df.isnull().sum().any()


# In[17]:


train_df.isnull().sum().head(30)


# In[18]:


train_df.isnull().sum().tail(50)


# Attrinutes of train_df:

# In[19]:


train_df.shape   # train dataset has 27321 rows and 80 columns.


# In[20]:


train_df.size


# Columns : "BLOCKID" is removing from train_df because it has all null value.
# 

# #Now chekcing varince for train dataset

# In[21]:


train_df.var()


# In[22]:


train_var = train_df.nunique()
col_to_drop = train_var[train_var ==1].index


# In[23]:


col_to_drop


# So we could infer from above that columns: "SUMLEVEL,"primary" has zero variance so we remove these two columns from train dataset.

# In[24]:


train_df_var = train_df.drop(["BLOCKID","primary","SUMLEVEL"],axis=1,inplace=True)


# In[25]:


train_df_var


# In[26]:


train_df.head(30)


# In[27]:


train_df[30:100]


# # Now treatment of null value with reason

# In[28]:


train_df.isnull().sum()[10:80]


# In[29]:


len(train_df.columns[train_df.isnull().sum(axis=0)> 0])


# so total 58 rows have null value
# 

# In[30]:


percent_missing = train_df. isnull(). sum() * 100 / len(train_df)


# In[31]:


percent_missing


# In[32]:


null_value_data = train_df[train_df.isnull().any(axis=1)]


# In[33]:


null_value_data


# So 77 rows and one row Blockid has null values. we have already drop Blockid so now we have total 77 rows null values left,now we calculate percentage to treat these null values, so what we do we divide null values columns 736 to total rows inot 100 to fidn out the percentage of null value in total dataset ,then we think weather this will make impact on dataset or not or we should fill these null value or not.

# In[34]:


round((736/27321)*100 ,2)


# So from above we could infer that we only have 2.69% data missing, we can safely delete these rows, without loosing much information from these row,so we do not fill these null values in dataset.
# 

# In[35]:


train_df.shape


# In[36]:


train_df = pd.concat([train_df, null_value_data, null_value_data]).drop_duplicates(keep=False)


# In[37]:


train_df


# so now we check the shape of the dataset after removing duplicate rows

# In[38]:


train_df.shape


# In[39]:


train_df.isnull().sum().any()


# # So from above we could infer that we dont have any null value in our dataset.

# In[40]:


train_df.size


# In[41]:


train_df.describe()


# so from above we could get statastic values of train dataset.

# In[42]:


train_df.dtypes[0:79]


# # Explonatory Data Analysis

# 4. Perform debt analysis. You may take the following steps:
# 
# 
# a) Explore the top 2,500 locations where the percentage of households with a second mortgage is the highest and percent ownership is above 10 percent. Visualize using geo-map.
# You may keep the upper limit for the percent of households with a second mortgage to 50 percent..
# 

# In[43]:


train_df.nlargest(2500,["second_mortgage","pct_own"])


# so above are the top 2500 location of household with second_mortgage.

# In[44]:


top_2500 = train_df[['state', 'lat', 'lng', 'second_mortgage', 'pct_own', 'place', 'state', 'city', 'COUNTYID', 'STATEID', 'home_equity', 'home_equity_second_mortgage', 'debt', 'hi_median', 'family_median']].nlargest(2563, ['second_mortgage', 'pct_own'])


# In[45]:


top_2500


# In[46]:


top_2500.pct_own.nunique()


# In[47]:


top_2500.pct_own.unique


# In[48]:


train_df[train_df.pct_own>0.1]   # this is the percentage ownership is above 10 % .


# In[49]:


train_df[train_df.pct_own>0.1].tail(50)


# In[50]:


train_df[train_df.pct_own>0.1][1180:2000]


# In[51]:


train_df[train_df.pct_own>0.1][800:1100]


# # Now we have to visualize the above data on geomap.

# In[52]:


pip install pyshp


# In[53]:


pip install plotly


# In[54]:


import plotly.graph_objects as go
import plotly.figure_factory as ff


# In[55]:


scope = ["USA"]

values = top_2500['second_mortgage'].tolist()

place = top_2500['place'].tolist()


# In[56]:


def zero_prefix(str_list):
    ''' prefixing 0's to numbers. Define the target length of your final number
     Function will add required no. of 0's to meet the target length''' 
        
    str_list = list(map(str, str_list))
    
    target_length = int(input("Enter Target Length of String: "))
    
    for i in range(len(str_list)):
        if len(str_list[i]) < target_length:
            str_list[i] = (target_length - len(str_list[i])) * '0'+ str_list[i]
    
    return str_list       

        #elif len(str_list[i]) <= 1:
                #str_list[i] = '00'+ str_list[i]


# In[57]:


top_2500.head(50)


# In[58]:


from bokeh.io import output_file, output_notebook, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, LogColorMapper, BasicTicker, ColorBar,
    DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)

from bokeh.plotting import gmap

from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.palettes import Viridis5


# In[59]:


map_options = GMapOptions(lat=37.88, lng=-122.23, map_type="roadmap", zoom=6)

plot = gmap( "AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY" , map_options=map_options,
                        title = 'Top 2500 Locations'
)

# source = ColumnDataSource(
#     data=dict(lat=[ 30.29,  30.20,  30.29],
#               lon=[-97.70, -97.74, -97.78])
# )

# p.circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, source=source)

# show(p)

source = ColumnDataSource(
    data=dict(
        lat=top_2500.lat.tolist(),
        lon=top_2500.lng.tolist(),
        size=top_2500.second_mortgage.tolist(),
        color=top_2500.pct_own.tolist()
    )
)
max_pct_own = top_2500.loc[top_2500['pct_own'].idxmax()]['pct_own']
min_pct_own = top_2500.loc[top_2500['pct_own'].idxmin()]['pct_own']

#color_mapper = CategoricalColorMapper(factors=['hi', 'lo'], palette=[RdBu3[2], RdBu3[0]])
#color_mapper = LogColorMapper(palette="Viridis5", low=min_median_house_value, high=max_median_house_value)
color_mapper = LinearColorMapper(palette=Viridis5)

circle = Circle(x="lon", y="lat", size="size", fill_color={'field': 'color', 'transform': color_mapper}, fill_alpha=0.5, line_color=None)
plot.add_glyph(source, circle)

color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))
plot.add_layout(color_bar, 'right')

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
#output_file("gmap_plot.html")
output_notebook()

show(plot)


# # b) Use the following bad debt equation:
# 
# Bad Debt = P (Second Mortgage ∩ Home Equity Loan)
# Bad Debt = second_mortgage + home_equity - home_equity_second_mortgage
# 

# In[60]:


top_2500['Bad_Debt'] = top_2500['second_mortgage'] + top_2500['home_equity'] - top_2500['home_equity_second_mortgage']
top_2500['Good_Debt'] = top_2500['debt'] - top_2500['Bad_Debt']


# In[61]:


top_2500['Good_Debt'] = top_2500['debt'] - top_2500['Bad_Debt']


# In[62]:


top_2500.head(80)


# In[63]:


top_2500[1500:2100]


# #  c) Now we have to create pie charts to show overall debt and bad debt.
# 

# In[64]:


size = 10
explode = [0.4] * size
explode = tuple(explode)
explode

explode_bd = [0.5] * size*2
explode_bd = tuple(explode_bd)
explode_bd

labels_D = ['GD', 'BD'] * size
labels_D = tuple(labels_D)
labels_D


# In[65]:


l1 = list(top_2500['Bad_Debt'] )
l1[:10]


# In[66]:


l2 = list(top_2500['Good_Debt'] )
l2[:10]


# In[67]:


l3 = sum(zip(l1, l2+[0]), ())
l3[:10]


# Now we have to create pie chart fro above 

# In[68]:


labels = list(top_2500.place[:10])
debt = list(top_2500.debt[:10])

sns.set_style("whitegrid")

gd_bd = l3[:20]

plt.figure(figsize = (15, 15))

color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
#color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

plt.pie(debt, labels = labels, startangle = 90, frame = True, radius =25, autopct='%1.1f%%', pctdistance=0.85, labeldistance = 0.9, colors = color_pal, explode = explode)
plt.pie(gd_bd, labels = labels_D, startangle = 90, frame = True, radius = 20, autopct='%1.1f%%', pctdistance=0.80,  labeldistance = 0.85, colors = color_pal, explode = explode_bd)
centre_circle = plt.Circle((0,0),15,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.tight_layout()


#  It is very difficult to show all 2500 locations, without compromising readability,so I have limited my selection to "Top 10" cities.
# 

#  # d) Now we ahve to create Box and whisker plot and analyze the distribution for 2nd mortgage, home equity, good debt, and bad debt for different cities.
# 

# In[69]:


second_mortgage = list(top_2500.second_mortgage)
home_equity = list(top_2500.home_equity)

Good_Debt = list(top_2500.Good_Debt)
Bad_Debt = list(top_2500.Bad_Debt)


# In[70]:


top_2500['city'].value_counts()[:30].index


# In[71]:


cities = ['Chicago', 'Los Angeles', 'Washington', 'Brooklyn',
                  'Milwaukee', 'Aurora', 'Jacksonville', 'Denver', 'Charlotte',
                  'Las Vegas', 'Bronx', 'Baltimore', 'Minneapolis',
                  'Cincinnati', 'Long Beach', 'Colorado Springs', 'Sacramento',
                  'San Diego', 'New Orleans', 'Columbus', 'Lowell', 'Orlando',
                  'Portland', 'San Jose', 'Alexandria', 'Dallas', 'Atlanta',
                  'Littleton', 'Miami', 'Oakland', 'Houston']


# In[72]:


cities


# In[73]:


boxplot_df = top_2500[top_2500['city'].isin (cities)]


# In[74]:


boxplot_df


# In[75]:


sns.set_style("whitegrid")

plt.figure(figsize = (25, 15))
sns.boxplot(x='city',y='second_mortgage',data=boxplot_df,palette='rainbow', order = ['Chicago', 'Los Angeles', 'Washington', 'Brooklyn',
                  'Milwaukee', 'Aurora', 'Jacksonville', 'Denver', 'Charlotte',
                  'Las Vegas', 'Bronx', 'Baltimore', 'Minneapolis',
                  'Cincinnati', 'Long Beach', 'Colorado Springs', 'Sacramento',
                  'San Diego', 'New Orleans', 'Columbus', 'Lowell', 'Orlando',
                  'Portland', 'San Jose', 'Alexandria', 'Dallas', 'Atlanta',
                  'Littleton', 'Miami', 'Oakland', 'Houston']).set_title('Second Mortgage distribution by cities', fontsize = 20)
plt.show()


# In[76]:


sns.set_style("whitegrid")

plt.figure(figsize = (25, 15))
sns.boxplot(x='city',y='home_equity',data=boxplot_df,palette='rainbow', order = ['Chicago', 'Los Angeles', 'Washington', 'Brooklyn',
                  'Milwaukee', 'Aurora', 'Jacksonville', 'Denver', 'Charlotte',
                  'Las Vegas', 'Bronx', 'Baltimore', 'Minneapolis',
                  'Cincinnati', 'Long Beach', 'Colorado Springs', 'Sacramento',
                  'San Diego', 'New Orleans', 'Columbus', 'Lowell', 'Orlando',
                  'Portland', 'San Jose', 'Alexandria', 'Dallas', 'Atlanta',
                  'Littleton', 'Miami', 'Oakland', 'Houston']).set_title('Home Equity distribution by cities', fontsize = 20)
plt.show()


# In[77]:


sns.set_style("whitegrid")

plt.figure(figsize = (25, 15))
sns.boxplot(x='city',y='Good_Debt',data=boxplot_df,palette='rainbow', order = ['Chicago', 'Los Angeles', 'Washington', 'Brooklyn',
                  'Milwaukee', 'Aurora', 'Jacksonville', 'Denver', 'Charlotte',
                  'Las Vegas', 'Bronx', 'Baltimore', 'Minneapolis',
                  'Cincinnati', 'Long Beach', 'Colorado Springs', 'Sacramento',
                  'San Diego', 'New Orleans', 'Columbus', 'Lowell', 'Orlando',
                  'Portland', 'San Jose', 'Alexandria', 'Dallas', 'Atlanta',
                  'Littleton', 'Miami', 'Oakland', 'Houston']).set_title('Good Debt distribution by cities', fontsize = 20)
plt.show()


# In[78]:


sns.set_style("whitegrid")

plt.figure(figsize = (25, 15))
sns.boxplot(x='city',y='Bad_Debt',data=boxplot_df,palette='rainbow', order = ['Chicago', 'Los Angeles', 'Washington', 'Brooklyn',
                  'Milwaukee', 'Aurora', 'Jacksonville', 'Denver', 'Charlotte',
                  'Las Vegas', 'Bronx', 'Baltimore', 'Minneapolis',
                  'Cincinnati', 'Long Beach', 'Colorado Springs', 'Sacramento',
                  'San Diego', 'New Orleans', 'Columbus', 'Lowell', 'Orlando',
                  'Portland', 'San Jose', 'Alexandria', 'Dallas', 'Atlanta',
                  'Littleton', 'Miami', 'Oakland', 'Houston']).set_title('Bad Debt distribution by cities', fontsize = 20)
plt.show()


# Here it is very difficult to show all 2500 locations, without compromising readability,so I have limited my selection to "Top 30" cities.
# 

#  # e) Now we have to create a collated income distribution chart for family income, house hold income, and remaining income.
# 

# In[79]:


top_2500['remaining_income'] = top_2500['family_median'] - top_2500['hi_median']


# In[80]:


income_chart = round(top_2500[['city', 'hi_median', 'family_median', 'remaining_income']], 2)
income_chart


# In[81]:


sns.set_style("whitegrid")
plt.figure(figsize = (8, 15))
sns.boxplot(data=top_2500[['family_median', 'hi_median', 'remaining_income']], palette=color_pal).set_title('Collated Income distribution', fontsize = 20)
plt.show()


# # Week-2 task

# 1. Now perform EDA and come out with insights into population density and age. You may have to derive new fields (make sure to weight averages for accurate measurements:
# 

# a) Use pop and ALand variables to create a new field called population density.
# 
# b) Use male_age_median, female_age_median, male_pop, and female_pop to create a new field called median age.
# 
# c) Visualize the findings using appropriate chart type
# 

# In[82]:


train_df.head(50)


# In[83]:


Density_df_eda = train_df[['state', 'city', 'place', 'ALand', 'pop', 'male_age_median', 'female_age_median', 'male_pop', 'female_pop']]


# In[84]:


Density_df_eda.head(50)


# In[85]:


Density_df_eda['pop_density'] = Density_df_eda['pop'] / Density_df_eda['ALand']


# In[86]:


Density_df_eda.head(30)


# In[87]:


Density_df_eda['median_age'] = (Density_df_eda['male_age_median'] *  Density_df_eda['male_pop'] + Density_df_eda['female_age_median'] *  Density_df_eda['female_pop'])  / Density_df_eda['pop']
Density_df_eda.head()


# In[88]:


Density_df_eda.nlargest(300, 'pop_density')


# In[89]:


sns.set_style("whitegrid")
plt.figure(figsize = (30, 15))
sns.boxplot(x = 'place', y = 'pop_density', data=Density_df_eda.nlargest(26585, 'pop_density'), palette=color_pal, order = ['New York City',
 'Mount Vernon City',
 'Pelham Manor',
 'Harbor Hills',
 'Sausalito City',
 'Chicago City',
 'Bellerose Terrace',
 'Lincolnwood',
 'Evanston City',
 'Halawa',
 'Guttenberg',
 'West Hollywood City',
 'West New York',
 'Daly City City',
 'Chelsea City',
 'Washington City',
 "Bailey's Crossroads",
 'Union City City',
 'Urban Honolulu',
 'Colwyn',
 'Hoboken City',
 'San Rafael City',
 'Yonkers City',
 'Jersey City City',
 'Boston City'])
plt.show()


# In[90]:


list(Density_df_eda.nsmallest(450, 'pop_density').state.unique())


# In[91]:


sns.set_style("whitegrid")
plt.figure(figsize = (25, 10))
sns.boxplot(x = 'state', y = 'pop_density', data=Density_df_eda.nlargest(26585, 'pop_density'), palette=color_pal, order = ['New York', 'California', 'Illinois', 'Hawaii', 'New Jersey', 'Massachusetts', 'District of Columbia', 'Virginia',
                                                                                                                                                                            'Pennsylvania', 'Florida', 'Puerto Rico', 'Maryland', 'Connecticut', 'Washington', 'Colorado', 'Wisconsin',
                                                                                                                            'Delaware', 'Oregon', 'Texas']).set_title('Population Density Distribution of THICKLY populated States', fontsize = 20)
plt.show()


# In[92]:


sns.set_style("whitegrid")
plt.figure(figsize = (25, 10))
sns.boxplot(x = 'state', y = 'pop_density', data=Density_df_eda.nsmallest(26585, 'pop_density'), palette=color_pal, order = ['Alaska', 'Montana', 'Utah', 'Oregon', 'Nevada', 'Colorado', 'Idaho', 'California', 'New Mexico',
                                                                                                                                                                                 'Maine', 'South Dakota', 'Wyoming', 'Nebraska', 'Texas', 'Kansas', 'North Dakota', 'Arizona',
                                                                                                                                                                                 'Washington', 'New York', 'Oklahoma', 'Minnesota', 'Louisiana', 'Michigan', 'Florida', 'Wisconsin', 'Mississippi',
                                                                                                                                                                                 'New Hampshire', 'Georgia', 'Missouri', 'Virginia', 'Alabama', 'Arkansas']).set_title('Population Density Distribution of THINLY populated States', fontsize = 20)
plt.show()


# In[93]:


sns.set_style("whitegrid")
plt.figure(figsize = (40, 15))
sns.boxplot(x = 'place', y = 'pop_density', data=Density_df_eda[Density_df_eda['state'] == 'New York'].nlargest(26585, 'pop_density'), palette=color_pal, order = ['New York City',
 'Mount Vernon City',
 'Pelham Manor',
 'Harbor Hills',
 'Bellerose Terrace',
 'Yonkers City',
 'Inwood',
 'South Valley Stream',
 'Ithaca City',
 'Saddle Rock',
 'Kings Point',
 'North Valley Stream',
 'Long Beach City',
 'Hempstead',
 'University Gardens',
 'New Rochelle City',
 'Elmont',
 'Great Neck Plaz']
).set_title('Population Density Distribution of Top 19 Cities of NEW YORK State', fontsize = 20)
plt.show()


# In[94]:


sns.set_style("whitegrid")
plt.figure(figsize = (35, 15))
sns.boxplot(x = 'place', y = 'pop_density', data=Density_df_eda[Density_df_eda['state'] == 'Alaska'].nlargest(26585, 'pop_density'), palette=color_pal, order = ['Anchorage', 'Point Mackenzie', 'Fairbanks City', 'College', 'Hope', 'South Van Horn', 
                                                                                                                                                                                    'Wasilla City', 'Juneau City', 'Ketchikan City', 'Sitka City', 'Lakes', 'Kodiak City', 'Badger', 'Gateway', 'North Pole City', 'Chena Ridge', 'Meadow Lakes']
).set_title('Population Density Distribution of Top 17 Cities of ALASKA State', fontsize = 20)
plt.show()


# In[95]:


print(list(Density_df_eda.nlargest(450, 'median_age').state.unique()))
print(len(list(Density_df_eda.nlargest(450, 'median_age').state.unique())))


# In[96]:


sns.set_style("whitegrid")
plt.figure(figsize = (35, 8))


ax = sns.boxplot(x = 'state', y = 'median_age', data=Density_df_eda.nlargest(26585, 'median_age'), palette=color_pal, 
            order = ['New York', 'Florida', 'California', 'Maryland', 'New Jersey', 'Arizona', 'Nevada', 'Arkansas', 'Illinois', 'North Carolina', 'South Carolina', 'Delaware', 'Ohio', 'Texas', 'Georgia', 'Alabama', 'New Mexico', 'Tennessee', 
            'Oregon', 'Michigan', 'Hawaii', 'Massachusetts', 'Pennsylvania', 'Minnesota', 'Wisconsin', 'Missouri', 'Washington', 'Colorado', 'Virginia', 'Maine', 'Mississippi', 'Louisiana', 'Indiana', 'Oklahoma']
           ).set_title('Median Age Distribution of  States with Older Population', fontsize = 20)

#ax.set(ylim=(0, 100))

plt.show()


# In[97]:


print(list(Density_df_eda.nsmallest(150, 'median_age').state.unique()))
print(len(list(Density_df_eda.nsmallest(150, 'median_age').state.unique())))


# In[98]:


sns.set_style("whitegrid")
plt.figure(figsize = (35, 12))
ax = sns.boxplot(x = 'state', y = 'median_age', data=Density_df_eda.nsmallest(26585, 'median_age'), palette=color_pal, 
            order = ['New York', 'Ohio', 'Georgia', 'New Jersey', 'Texas', 'Colorado', 'Virginia', 'California', 'Mississippi', 'Oregon', 'Arizona', 'Utah', 'Pennsylvania', 'Michigan', 'Arkansas', 'Maine', 'Florida', 'North Carolina', 
                     'Minnesota', 'Iowa', 'Connecticut', 'Maryland', 'Wisconsin', 'Louisiana', 'Alabama', 'Massachusetts', 'Washington', 'Illinois', 'Tennessee', 'Indiana', 'District of Columbia', 'South Carolina', 'Oklahoma', 'Delaware']
           ).set_title('Median Age Distribution of  States with Younger Population', fontsize = 20)
#ax.set(ylim=(0, 100))
plt.show()


# In[99]:


sns.set_style("whitegrid")
plt.figure(figsize = (35, 12))
ax = sns.boxplot(x = 'city', y = 'median_age', data=Density_df_eda[Density_df_eda['state'] == 'New York'].nlargest(26585, 'median_age'), palette=color_pal, 
            order =['Bellerose', 'Ridge', 'Somers', 'Bronx', 'Yorktown Heights', 'Johnson City', 'Brooklyn', 'Williamsville', 'Fishers Island', 'Lake George', 'Chestertown', 'Rhinebeck', 
                    'Southampton', 'Richmond Hill', 'Mount Vernon', 'Jamaica', 'Long Lake', 'Melville', 'Amityville', 'East Rochester', 'Rome', 'Calverton', 'Woodstock', 'White Plains', 'Craryville', 
                    'Rochester', 'Youngstown', 'Sag Harbor', 'Valatie', 'Yonkers', 'Hammondsport', 'Utica', 'Middle Island', 'New York', 'Staten Island', 'Caroga Lake', 'Willsboro']
)
ax.set_title('Median Age Distribution of  cities in New York State', fontsize = 35)
ax.set(ylim=(15, 75))
plt.show()


# In[100]:


sns.set_style("whitegrid")
plt.figure(figsize = (35, 10))
ax = sns.boxplot(x = 'city', y = 'median_age', data=Density_df_eda[Density_df_eda['state'] == 'Alaska'].nlargest(26585, 'median_age'), palette=color_pal, 
            order =['Haines', 'Talkeetna', 'Anchorage', 'Hoonah', 'Rampart', 'Moose Pass', 'Ketchikan', 'Juneau', 'Port Graham', 'Klawock', 'Skagway', 'Nikiski', 'Yakutat', 'Sitka', 'Fairbanks', 
                    'Eagle River', 'Healy', 'Kodiak', 'Larsen Bay', 'Wasilla', 'Palmer', 'Kasilof', 'Copper Center', 'Fort Yukon']
)
ax.set_title('Median Age Distribution of  cities in Alaska State', fontsize = 30)
ax.set(ylim=(20,50))
plt.show()


# In[101]:


list(Density_df_eda[Density_df_eda['state'] == 'New York'].nlargest(600, 'pop_density').place.unique())
print(len(list(Density_df_eda[Density_df_eda['state'] == 'New York'].nlargest(600, 'pop_density').place.unique())))


# In[102]:


print(list(Density_df_eda[Density_df_eda['state'] == 'Alaska'].nlargest(42, 'median_age').city.unique()))
print(len(list(Density_df_eda[Density_df_eda['state'] == 'Alaska'].nlargest(42, 'median_age').city.unique())))


# In[103]:


train_df.head(50)


# # 2.Create bins for population into a new variable by selecting appropriate class interval so that the number of categories don’t exceed 5 for the ease of analysis.
# 
# a) Analyze the married, separated, and divorced population for these population brackets.
# 
# 
# b) Visualize using appropriate cmhart type.
# 

# In[104]:


age_df = train_df[['state', 'city', 'place', 'pop', 'male_pop', 'female_pop', 'male_age_median', 'female_age_median', 'married', 'separated', 'divorced']]


# In[105]:


age_df[40:400]


# In[106]:


train_df.male_age_median.unique()


# we could infer from above that The IntelliSense Age Group defaults are:
# 
# Youth (<18)
# 
# Young Adult (18 to 35)
# 
# Adult (36 to 55)
# 
# Senior (56 and up)
# 
# 

# In[107]:


bins = [0, 12,18, 35, 55, 100]
labels = ['kids', 'Youth', 'Young Adult', 'Adult', 'Senior']


# df['binned'] = pd.cut(df['percentage'], bins, labels = labels)
# 

# In[108]:


age_df['male_population_bracket'] = pd.cut(age_df['male_age_median'], bins, labels = labels)


# In[109]:


age_df["male_population_bracket"]


# In[110]:


age_df['female_population_bracket'] = pd.cut(age_df['female_age_median'], bins, labels = labels)


# In[111]:


age_df['female_population_bracket']


# In[112]:


age_df.head(40)


# In[113]:


sns.set_style("whitegrid")

plt.figure(figsize = (25, 15))

ax = sns.barplot(x = 'state', y = 'married', hue = 'male_population_bracket', data = age_df, palette=color_pal, 
           order = ['New York', 'Ohio', 'Georgia', 'New Jersey', 'Texas', 'Colorado', 'Virginia', 'California', 'Mississippi', 'Oregon', 'Arizona', 'Utah', 'Pennsylvania', 'Michigan', 'Arkansas', 'Maine', 'Florida', 'North Carolina', 
                     'Minnesota', 'Iowa', 'Connecticut', 'Maryland', 'Wisconsin', 'Louisiana', 'Alabama', 'Massachusetts', 'Washington', 'Illinois', 'Tennessee', 'Indiana', 'District of Columbia', 'South Carolina', 'Oklahoma', 'Delaware'])

ax.set_title('Married Male population by state', fontsize = 25)

plt.show()


# # What we have found from above that "Georgia","Ohio" have married male kids.

# In[114]:


age_df.city.unique()   # For cities


# In[115]:


sns.set_style("whitegrid")
plt.figure(figsize = (25, 10))

ax = sns.barplot(x = 'state', y = 'married', hue = 'female_population_bracket', data = age_df, palette=color_pal, 
           order = ['New York', 'Ohio', 'Georgia', 'New Jersey', 'Texas', 'Colorado', 'Virginia', 'California', 'Mississippi', 'Oregon', 'Arizona', 'Utah', 'Pennsylvania', 'Michigan', 'Arkansas', 'Maine', 'Florida', 'North Carolina', 
                     'Minnesota', 'Iowa', 'Connecticut', 'Maryland', 'Wisconsin', 'Louisiana', 'Alabama', 'Massachusetts', 'Washington', 'Illinois', 'Tennessee', 'Indiana', 'District of Columbia', 'South Carolina', 'Oklahoma', 'Delaware'])

ax.set_title('Married Female population by state', fontsize = 25)

plt.show()


# So from above we could infer the married female population by state.

# 
# 
# # So from abovewe could iner that only "Newyork" has married female kids or youth, NO other state has married female kidy or Youth.
# 

# In[116]:


sns.set_style("whitegrid")
plt.figure(figsize = (35, 15))

ax = sns.barplot(x = 'state', y = 'separated', hue = 'male_population_bracket', data = age_df, palette=color_pal, 
           order = ['New York', 'Ohio', 'Georgia', 'New Jersey', 'Texas', 'Colorado', 'Virginia', 'California', 'Mississippi', 'Oregon', 'Arizona', 'Utah', 'Pennsylvania', 'Michigan', 'Arkansas', 'Maine', 'Florida', 'North Carolina', 
                     'Minnesota', 'Iowa', 'Connecticut', 'Maryland', 'Wisconsin', 'Louisiana', 'Alabama', 'Massachusetts', 'Washington', 'Illinois', 'Tennessee', 'Indiana', 'District of Columbia', 'South Carolina', 'Oklahoma', 'Delaware'])

ax.set_title('Separated Male population by state', fontsize = 25)

plt.show()


# "Ohio", has Largest number of Divorced Male KIDS.
# 
# "Connecticut", has Largest number of Divorced Male YOUTH.
# 
# "Maine, Indiana & Oklahoma", has Largest number of Divorced Male YOUNG ADULTS
# 
# "Arkansas, Maine, Indiana & Oklahoma", has Largest number of Divorced Male ADULTS
# 
# "Louisiana & OKlahoma", has Largest number of Divorced Male SENIORS.
# 
# Looks like "OKlahoma", is the Divorce Capital for MALE population.
# 

# In[117]:


sns.set_style("whitegrid")

plt.figure(figsize = (35, 15))

ax = sns.barplot(x = 'state', y = 'divorced', hue = 'female_population_bracket', data = age_df, palette=color_pal, 
           order = ['New York', 'Ohio', 'Georgia', 'New Jersey', 'Texas', 'Colorado', 'Virginia', 'California', 'Mississippi', 'Oregon', 'Arizona', 'Utah', 'Pennsylvania', 'Michigan', 'Arkansas', 'Maine', 'Florida', 'North Carolina', 
                     'Minnesota', 'Iowa', 'Connecticut', 'Maryland', 'Wisconsin', 'Louisiana', 'Alabama', 'Massachusetts', 'Washington', 'Illinois', 'Tennessee', 'Indiana', 'District of Columbia', 'South Carolina', 'Oklahoma', 'Delaware'])

ax.set_title('Divorced Female population by state', fontsize = 25)

plt.show()


# "Maine", has Largest number of Divorced Female YOUNG ADULTS
# 
# "Maine", has Largest number of Divorced Female ADULTS
# 
# "Louisiana", has Largest number of Divorced Female SENIORS.
# 
# AND "Newyork", is the only state that has Divorced Female YOUTH.
# 
# 

# 
# # 3. Please detail the observations for rent as a percentage of income at an overall level, and for different states.
# 

# In[118]:


train_df.head()


# In[119]:


rent_df = train_df[['state', 'city', 'rent_median', 'hi_median', 'family_median']]


# In[120]:


rent_df.head(40)


# In[121]:


Overall_rent_percentage = (rent_df['rent_median'].sum() / rent_df['hi_median'].sum()) * 100
round(Overall_rent_percentage, 2)


# # So we could infer from above that the overall Rent as a percentage of Overall House Hold Income is around 1.74%.
# 

# In[122]:


rent_df['ov_rent_pcnt'] = round((rent_df['rent_median'] / rent_df['hi_median']) * 100, 2)


# In[123]:


rent_df.head()


# In[124]:


print(list(rent_df.nlargest(500, 'ov_rent_pcnt').state.unique()))
print(len(list(rent_df.nlargest(500, 'ov_rent_pcnt').state.unique())))


# In[125]:


sns.set_style("whitegrid")
plt.figure(figsize = (25, 15))
ax = sns.boxplot(x = 'state', y = 'ov_rent_pcnt', data=rent_df.nlargest(26585, 'ov_rent_pcnt'), palette=color_pal, 
            order = ['Georgia', 'Texas', 'California', 'New York', 'Florida', 'Washington', 'Oregon', 'Pennsylvania', 'Maryland', 'Virginia', 'Mississippi', 'Alabama', 'Michigan', 'Louisiana', 
                     'Iowa', 'Puerto Rico', 'New Jersey', 'Illinois', 'Arizona', 'North Carolina', 'South Carolina', 'Tennessee', 'Ohio', 'Wisconsin', 'Missouri', 'Connecticut', 'Minnesota', 
                     'Massachusetts', 'Indiana', 'Colorado', 'Kansas', 'Oklahoma', 'District of Columbia', 'New Mexico', 'Hawaii', 'Maine', 'Arkansas', 'Vermont', 'Rhode Island', 'Kentucky']
           ).set_title('Rent as percentage of House Hold Income by State', fontsize = 30)
#ax.set(ylim=(0, 100))
plt.show()


# 4.To Perform correlation analysis for all the relevant variables by creating a heatmap. Describe your findings.
# 

# In[126]:


sns.set_style("whitegrid")

corr = train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

kot = corr[corr>=.6]
plt.figure(figsize=(35,25))
sns.heatmap(kot, cmap="Blues", annot = True, mask = mask, linewidths=1, linecolor='red').set_title('Positive correlation Heat Map', fontsize = 20)
plt.grid('on', )
plt.show()


# So from above we could iner that - "Population parameters" have Strong positive correlation wih "Sample Parameters".
# 

#                                                                                          - "Male Population is highly correlated with Female population. <br/>      
#                                                                                              <br/>   
#                                                                                          - "rent Mean & Median" has  high positive correlation with "House hold income Mean, Median and Standard Deviation",  <br/>
#                                                                                                 <br/>
#                                                                                             where as "rent Standard Deviation has positive correlatioin with "hc mortgage mean & median". <br/>       
#                                                                                                <br/> 
#                                                                                          - "House hold income and Family income are highly positively correlated. <br/>        
#                                                                                                <br/> 
#                                                                                          - "Family Income"  and "hc_mortgage" are positively correlated. <br/>      
#                                                                                                <br/> 
#                                                                                          - "pct_own" is positively correlated with "Married" marital status  </h1>
# 

# In[127]:


sns.set_style("whitegrid")
kot = corr[corr <=-.3]
plt.figure(figsize=(55,25))
sns.heatmap(kot, cmap="Reds", annot = True, mask = mask, linewidths=1, linecolor='red').set_title('Negative correlation Heat Map', fontsize = 20)
plt.grid('on', )
plt.show()


# So we could infer thatthe  "House hold income and Family Income" has Strong negative correlation with ["married_snp", "separated", "divorced"].
# 

#                                                                                          - "High School Degree in both "Males and Females" have Strong negative correlation with ["married_snp", "separated"] <br/>      
#                                                                                              <br/>   
#                                                                                          - "pct_own" has Strong negative correlation with  ["married_snp", "separated"]  <br/>
#                                                                                                 <br/>
#                                                                                          - "hi_median" has Strong negative correlation with "rent_gt_30", indicating that most households look for properties with rent less than 30% of their house hold income.. <br/>        
# </h1>
# 
# 

#  # Data Pre-processing:
#     
#  # Project Task: Week 3
#  
# 
# 1. The economic multivariate data has a significant number of measured variables. The goal is to find where the measured variables depend on a number of smaller unobserved common factors or latent variables.
# Each variable is assumed to be dependent upon a linear combination of the common factors, and the coefficients are known as loadings. Each measured variable also includes a component due to independent random variability, known as “specific variance” because it is specific to one variable. Obtain the common factors and then plot the loadings. Use factor analysis to find latent variables in our dataset and gain insight into the linear relationships in the data. Following are the list of latent variables:
# 
# • Highschool graduation rates
# • Median population age
# • Second mortgage statistics
# • Percent own
# • Bad debt expense
# 

# In[128]:


train_df.info()


# In[129]:


train_df['Bad_Debt'] = train_df['second_mortgage'] + train_df['home_equity'] - train_df['home_equity_second_mortgage']


# In[130]:


train_df["Bad_Debt"]


# In[131]:


for col in train_df.columns:
    print(col,' = ' ,train_df[col].dtype)


# In[132]:


def cat_variables(df):
    cat_variables = list(df.select_dtypes(exclude = ['int', 'float']).columns)
    return cat_variables


# In[133]:


def num_variables(df):
    num_variables = list(df.select_dtypes(include = ['int', 'float']).columns)
    return num_variables


# In[134]:


train_df.city.dtype


# In[135]:


cat_variables(train_df)


# In[136]:


num_variables(train_df)


# In[137]:


fa_train_df = train_df[num_variables(train_df)]
fa_train_df


# In[138]:


# exclude columns you don't want
fa_train_df = fa_train_df[fa_train_df.columns[~fa_train_df.columns.isin(['SUMLEVEL', 'lat', 'lng', 
                                                                                                            'ALand', # 'AWater'
                                                                                                        ])]]


# In[139]:


pip install factor_analyzer


# In[140]:


from factor_analyzer import FactorAnalyzer
import warnings
warnings.filterwarnings('ignore')


# In[141]:


# TO create factor analysis object and perform factor analysis

fa = FactorAnalyzer( rotation=None, n_factors = 25)
fa.fit(fa_train_df)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev


# In[142]:


print(sorted(ev, reverse=True))


# In[143]:


loadings = fa.loadings_


# In[144]:


xvals = range(1, fa_train_df.shape[1]+1)


# In[145]:


sns.set()
plt.figure(figsize = (25,10))
plt.scatter(xvals, ev)
plt.plot(xvals, ev)
plt.title('Scree plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
plt.grid(color = 'green', )
plt.grid(b=True, which='minor', color='r', linestyle='--')
plt.minorticks_on()
plt.show()


# In[146]:


Factors  = pd.DataFrame.from_records(loadings)

Factors = Factors.add_prefix('Factor ')

Factors.index = fa_train_df.columns
Factors


# In[147]:


fa = FactorAnalyzer( rotation="varimax", n_factors = 12)
fa.fit(fa_train_df)
loadings = fa.loadings_


# In[148]:


Factors  = pd.DataFrame.from_records(loadings)

Factors = Factors.add_prefix('Factor ')

Factors.index = fa_train_df.columns
Factors


#  • Highschool graduation rates
#  
#  • Median population age
# 
#  • Second mortgage statistics
# 
#  • Percent own
#  
#  • Bad debt expense
# 

# In[149]:


Factors_df = round(Factors.loc[['hs_degree', 'hs_degree_male', 'hs_degree_female',"male_age_median", "female_age_median", "home_equity_second_mortgage", 'second_mortgage', 'second_mortgage_cdf', 'pct_own', 'Bad_Debt'], :], 2)


# In[150]:


def color_negative_red(value):
  """
  Colors elements in a dateframe
  green if positive and red if
  negative. Does not color NaN
  values.
  """

  if value < -0.6:
    color = 'red'
  elif value > 0.6:
    color = 'green'
  else:
    color = 'black'

  return 'color: %s' % color


# In[151]:


Factors_df.style.applymap(color_negative_red)


# Looks "Related parameters" are loading on Unique Factors.
# 

# In[152]:


len(fa_train_df.columns)


# In[153]:


# Get variance of each factors
fact_variance  = fa.get_factor_variance()
fact_variance


# In[154]:


Factor_variance  = pd.DataFrame.from_records(fact_variance)

Factor_variance = Factor_variance.add_prefix('Factor ')

Factor_variance.index = ['SS Loadings', 'Proportion Var', 'Cumulative Var']
round(Factor_variance, 2)


# # Data Modeling Task
# 

# Project Task: Week 4
# 

# # 1. Build a linear Regression model to predict the total monthly expenditure for home mortgages loan.
# 

# - Please refer ‘deplotment_RE.xlsx’. Column hc_mortgage_mean is predicted variable. This is the mean monthly mortgage and owner costs of specified geographical location.
# 
# - Note: Exclude loans from prediction model which have NaN (Not a Number) values for hc_mortgage_mean.
# 
#     a) Run a model at a Nation level. If the accuracy levels and R square are not satisfactory proceed to below step
# 
#     b) Run another model at State level. There are 52 states in USA.
#     
#      Keep below considerations while building a linear regression model. Data Modeling :
#     
#     &nbsp&nbsp&nbsp&nbsp• Variables should have significant impact on predicting Monthly mortgage and owner costs
# 
#     &nbsp&nbsp&nbsp&nbsp• Utilize all predictor variable to start with initial hypothesis
# 
#     &nbsp&nbsp&nbsp&nbsp• R square of 60 percent and above should be achieved
# 
#     &nbsp&nbsp&nbsp&nbsp• Ensure Multi-collinearity does not exist in dependent variables
# 
#     &nbsp&nbsp&nbsp&nbsp• Test if predicted variable is normally distributed
# 
# 

# In[155]:


train_df1 = pd.read_csv("C:\\Users\\Samiksha\\Downloads\\Project_1\\Project 1\\train.csv")


# In[156]:


train_df1


# In[157]:


train_df1.isnull().sum()


# In[158]:


percent_missing1 = train_df1. isnull(). sum() * 100 / len(train_df1)


# In[159]:


percent_missing1


# In[160]:


null_value_data1 = train_df1[train_df1.isnull().any(axis=1)]


# In[161]:


null_value_data1


# In[162]:


train_df1.drop('BLOCKID', axis=1, inplace=True)


# In[163]:


test_df = pd.read_csv("C:\\Users\\Samiksha\\Downloads\\Project_1\\Project 1\\test.csv")


# In[164]:


test_df


# In[165]:


test_df.drop('BLOCKID', axis=1, inplace=True)


# In[166]:


train_df1.isnull().sum()


# In[167]:


test_df.isnull().sum()


# In[168]:


train_df1 = train_df1.dropna()
train_df1 = train_df1.reset_index(drop=True)


# In[169]:


test_df = test_df.dropna()
test_df = test_df.reset_index(drop=True)


# In[171]:


train_df1.shape


# In[172]:


test_df.shape


# In[174]:


train_df1[category_columns]


# In[175]:


train_df1[num_variables(train_df1)]


# In[176]:


train_df1.drop('SUMLEVEL', inplace = True, axis = 1)


# In[177]:


test_df.drop('SUMLEVEL', inplace = True, axis = 1)


# In[179]:


train_df1[num_variables(train_df1)]


# In[180]:


num_2_cat = ['UID','COUNTYID', 'STATEID', 'zip_code', 'area_code', 'lat', 'lng']


# In[181]:


train_df1.info()


# In[182]:


for col in num_2_cat:
    train_df1[col] = train_df1[col].astype('category')
    test_df[col] = test_df[col].astype('category')


# In[183]:


print(train_df1.info())
print('-----------')
print(test_df.info())


# In[184]:


train_df1[cat_variables(train_df1)]


# In[185]:


obj_2_cat = ['state', 'state_ab', 'city', 'place', 'type', 'primary']


# In[186]:


for col in obj_2_cat:
    train_df1[col] = train_df1[col].astype('category')
    test_df[col] = test_df[col].astype('category')


# In[187]:


train_df1.info()


# In[188]:


train_df1[['hc_mortgage_mean']]


# In[189]:


# Plot
kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

plt.figure(figsize=(15,5), dpi= 80)
sns.distplot(train_df.hc_mortgage_mean, color="dodgerblue", label="hc_mortgage_mean", **kwargs)
# sns.distplot(x2, color="orange", label="SUV", **kwargs)
# sns.distplot(x3, color="deeppink", label="minivan", **kwargs)
# plt.xlim(50,75)
plt.legend();


# # So here in this dataset the Target Variable "hc_mortgage_mean" has a Positive Skew.
# 

# In[191]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, SCORERS


# In[192]:


lr = LinearRegression()


# In[193]:


cat_cols_2_drop = ['UID', 'state', 'state_ab', 'city', 'place', 'type', 'primary', 'zip_code', 'area_code', 'lat', 'lng']


# In[194]:


train_df1.drop(cat_cols_2_drop, axis=1, inplace=True)


# In[195]:


test_df.drop(cat_cols_2_drop, axis=1, inplace=True)


# In[196]:


train_df1.drop(['COUNTYID', 'STATEID'], axis=1, inplace=True)


# In[197]:


test_y = test_df['hc_mortgage_mean']


# In[198]:


test_df.drop(['COUNTYID', 'STATEID', 'hc_mortgage_mean'], axis=1, inplace=True)


# In[199]:


print(train_df.shape, test_df.shape)


# In[200]:


train_X = train_df1.drop(columns=['hc_mortgage_mean'])
train_y = train_df1['hc_mortgage_mean']


# In[201]:


lr.fit(train_X, train_y)


# In[202]:


predict_train = lr.predict(train_X)
predict_test = lr.predict(test_df)


# In[204]:


# model evaluation for testing set

mae = mean_absolute_error(test_y, predict_test)
mse = mean_squared_error(test_y, predict_test)
r2 = r2_score(test_y, predict_test)

print("The model performance for test set")
print("--------------------------------------")
print('MAE is {}'.format(round(mae, 3)))
print('MSE is {}'.format(round(mse, 3)))
print('RMSE is {}'.format(round(mse**(0.5), 3)))
print('R2 score is {}'.format(round(r2, 3)))



# # Regression Model with all dependent numeric variables ,Country level is giving R SQUARED metric of 98.8%. ,So we skipping state level Regression Model
# 

# 
# 
# 
# 
# 
# Now we are finding the correlation

# In[205]:


correlated_features = set()
correlation_matrix = train_df1.drop('hc_mortgage_mean', axis=1).corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)


# In[206]:


correlated_features


# In[208]:


corr_list = ['debt_cdf', 'family_mean', 'family_median', 'family_sample_weight', 'family_samples', 'family_stdev', 'female_age_mean', 'female_age_median',
                     'female_age_sample_weight', 'female_age_samples', 'female_pop', 'hc_median', 'hc_mortgage_samples', 'hc_sample_weight', 'hi_median',
                     'hi_samples', 'hi_stdev', 'home_equity_cdf', 'hs_degree_female', 'hs_degree_male', 'male_age_median', 'male_age_sample_weight',
                 'male_age_samples', 'male_pop', 'rent_gt_25', 'rent_gt_30', 'rent_gt_35', 'rent_gt_40', 'rent_gt_50', 'rent_median', 'rent_samples', 'second_mortgage', 'universe_samples', 'used_samples']


# In[209]:


train_df1.drop(corr_list, axis=1, inplace=True)


# In[210]:


test_df.drop(corr_list, axis=1, inplace=True)


# In[211]:


print(train_df1.shape, test_df.shape)


# 
# 
# 
# # Dropped MultiCollinear variables and then we run the Regression Model.
# 

# In[212]:


train_df1.head()


# In[213]:


train_X = train_df1.drop(columns=['hc_mortgage_mean'])
train_y = train_df1['hc_mortgage_mean']


# In[214]:


lr.fit(train_X, train_y)


# In[215]:


predict_train = lr.predict(train_X)
predict_test = lr.predict(test_df)


# #  Model evaluation for test set
# 

# In[217]:



mae = mean_absolute_error(test_y, predict_test)
mse = mean_squared_error(test_y, predict_test)
r2 = r2_score(test_y, predict_test)

print("The model performance for test set")
print("--------------------------------------")
print('MAE is {}'.format(round(mae, 3)))
print('MSE is {}'.format(round(mse, 3)))
print('RMSE is {}'.format(round(mse**(0.5), 3)))
print('R2 score is {}'.format(round(r2, 3)))


# so we can see the RMSE,R2 SCORE MSE value from above for test set.

# In[219]:


sorted(SCORERS.keys())


# 
# 
# 
# 
# # So now we Check how close our algorithm is predicting, by passing the inputs from our test set and compare them to the target values.
# 

# In[220]:


import random
randomlist = []
for i in range(0,100):
    n = random.randint(1,len(test_df))
    randomlist.append(n)
print(randomlist)


# In[221]:


pre_out = []
out = []

for i in randomlist:
    data_in = [list(test_df.iloc[i])]
    pre_data_out = lr.predict(data_in)
    data_out = test_y .iloc[i]
    
    print(i, pre_data_out, data_out)
    
    pre_out.append(pre_data_out)
    out.append(data_out)


# In[222]:


pre_out


# so above we can check pre-out numbers array.

# In[223]:


x = [2,3,5,9,1,0,2,3]

def my_min(sequence):
    """return the minimum element of sequence"""
    low = sequence[0] # need to start with some value
    for i in sequence:
        if i < low:
            low = i
    return low

print(my_min(x))


# In[224]:


x = [2,3,5,9,1,0,2,3]

def my_maxi(sequence):
    """return the minimum element of sequence"""
    maxi = sequence[0] # need to start with some value
    for i in sequence:
        if i > maxi:
            maxi = i
    return maxi

print(my_maxi(x))


#  9 is the maximum sequence of sequence element and 0 is the minimum sequence element.

# # Now i am shwoing a graph of predicted and actual values subplot to get the visualization of predicted vs actual data value.

# In[225]:


fig, ax = plt.subplots(figsize=(35,18))
ax.scatter(pre_out, out, edgecolors=(0, 0, 1))
ax.plot([my_min(out), my_maxi(out)], [my_min(out), my_maxi(out)], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()


# Now model evaluete the model for test set

# In[226]:


mae = mean_absolute_error(test_y, predict_test)
mse = mean_squared_error(test_y, predict_test)
r2 = r2_score(test_y, predict_test)

print("The model performance for test set")
print("--------------------------------------")
print('MAE is {}'.format(round(mae, 3)))
print('MSE is {}'.format(round(mse, 3)))
print('RMSE is {}'.format(round(mse**(0.5), 3)))
print('R2 score is {}'.format(round(r2, 3)))


# # So from above we can the RMSE,R2SCORE AND MSE value of test dataset.

# # AND What i have have achieved an adjusted R Squared value of 98.8% which is pretty close to 1, indicating that our selected "Independent Variables" are highly correlated to our "Dependent Variable" and our model is able to predict very accurately in the dataset real estate.
# 

# In[ ]:




