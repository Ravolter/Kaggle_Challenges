#!/usr/bin/env python
# coding: utf-8

# In[153]:


import numpy as np, pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

#sklearn_pandas.CategoricalImputer

os.chdir("E:/2. Kaggle/Housing prices/home-data-for-ml-course")


# In[ ]:





# In[154]:


df_test = pd.read_csv("test.csv")
df_train = pd.read_csv("train.csv")


# In[155]:


df_train.head()


# In[156]:


df_train.shape


# In[157]:


df_train.describe()


# In[158]:


df_train.isnull().sum() > 0


# ### 3. EDA

# In[159]:


#histogram
df_train['SalePrice'].hist(bins = 40)


# In[160]:


cor_matrix = df_train.corr()
f, ax = plt.subplots(figsize=(30, 19))
sns.set(font_scale=2)
sns.heatmap(cor_matrix,square = True)


# In[161]:


features = cor_matrix.SalePrice.sort_values(ascending = False)[1:10].index


# In[162]:


sns.pairplot(df_train[features])
plt.show()


# In[163]:


df_train = df_train.drop(['Id'], axis=1)
df_test = df_test.drop(['Id'], axis = 1)


# In[164]:


testing_null = pd.isnull(df_test).sum()
training_null = pd.isnull(df_train).sum()


# In[165]:


null = pd.concat([testing_null,training_null],axis = 1, keys = ["testing","training"])
null


# In[166]:


#Based on the description data file provided, all the variables who have meaningfull Nan

null_with_meaning = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]


# In[167]:


for i in null_with_meaning:
    df_train[i].fillna("None",inplace = True)
    df_test[i].fillna("None",inplace = True)


# In[168]:


null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #few missing values
null_many


# In[173]:


df_train.info()


# In[174]:


df_train.drop("LotFrontage", axis=1, inplace=True)
df_test.drop("LotFrontage", axis=1, inplace=True)


# In[175]:


#I chose to use the mean function for replacement
df_train["GarageYrBlt"].fillna(df_train["GarageYrBlt"].mean(), inplace=True)
df_test["GarageYrBlt"].fillna(df_test["GarageYrBlt"].mean(), inplace=True)
#data["GarageYrBlt"].fillna(data["GarageYrBlt"].mean(), inplace=True)

df_train["MasVnrArea"].fillna(df_train["MasVnrArea"].mean(), inplace=True)
df_test["MasVnrArea"].fillna(df_test["MasVnrArea"].mean(), inplace=True)
#data["MasVnrArea"].fillna(data["MasVnrArea"].median(), inplace=True)

df_train["MasVnrType"].fillna("None", inplace=True)
df_test["MasVnrType"].fillna("None", inplace=True)
#data["MasVnrType"].fillna("None", inplace=True)


# In[176]:


types_train = df_train.dtypes #type of each feature in data: int, float, object
num_train = types_train[(types_train == int) | (types_train == float)] #numerical values are either type int or float
cat_train = types_train[types_train == object] #categorical values are type object

#we do the same for the test set
types_test = df_test.dtypes
num_test = types_test[(types_test == int) | (types_test == float)]
cat_test = types_test[types_test == object]


# ### Numerical Imputation

# In[177]:


numerical_values_train = list(num_train.index)
numerical_values_test = list(num_test.index)

fill_num = numerical_values_train + numerical_values_test
print(fill_num)


# In[178]:


for i in fill_num:
    df_train[i].fillna(df_train[i].mean(),inplace = True)
    df_test[i].fillna(df_test[i].mean(),inplace = True)


# ### Categorical Imputation

# In[179]:


df_train.shape, df_test.shape


# In[180]:


categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)


# In[181]:


fill_cat = []

for i in categorical_values_train:
    if i in list(null_few.index):
        fill_cat.append(i)
print(fill_cat)


# In[182]:


def most_common_term(data,feature):
    return data[feature].value_counts().index[0]


# In[183]:


most_common = []

for i in fill_cat:
    most_common.append(most_common_term(df_train,i))
    
most_common


# In[184]:


k = 0
for i in fill_cat:
    df_train.fillna(most_common[k], inplace=True)
    df_test.fillna(most_common[k], inplace=True)
    k += 1


# In[185]:


df_train.isnull().sum().sum()


# In[186]:


df_test.isnull().sum().sum()


# ### 4. Feature Engineering

# #### Sales price is skewed. So apply log transform

# In[187]:


df_train["log_sales_price"] = df_train["SalePrice"].apply(lambda x: np.log(x))


# In[188]:


df_train["log_sales_price"].hist(bins = 40)


# In[189]:


df_train_add = df_train.copy()

df_train_add['TotalSF']=df_train_add['TotalBsmtSF'] + df_train_add['1stFlrSF'] + df_train_add['2ndFlrSF']

df_train_add['Total_Bathrooms'] = (df_train_add['FullBath'] + (0.5 * df_train_add['HalfBath']) +
                               df_train_add['BsmtFullBath'] + (0.5 * df_train_add['BsmtHalfBath']))

df_train_add['Total_porch_sf'] = (df_train_add['OpenPorchSF'] + df_train_add['3SsnPorch'] +
                              df_train_add['EnclosedPorch'] + df_train_add['ScreenPorch'] +
                              df_train_add['WoodDeckSF'])

df_test_add = df_test.copy()

df_test_add['TotalSF']=df_test_add['TotalBsmtSF'] + df_test_add['1stFlrSF'] + df_test_add['2ndFlrSF']

df_test_add['Total_Bathrooms'] = (df_test_add['FullBath'] + (0.5 * df_test_add['HalfBath']) +
                               df_test_add['BsmtFullBath'] + (0.5 * df_test_add['BsmtHalfBath']))

df_test_add['Total_porch_sf'] = (df_test_add['OpenPorchSF'] + df_test_add['3SsnPorch'] +
                              df_test_add['EnclosedPorch'] + df_test_add['ScreenPorch'] +
                              df_test_add['WoodDeckSF'])


# In[190]:


## For ex, if PoolArea = 0 , Then HasPool = 0 too

df_train_add['haspool'] = df_train_add['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_train_add['has2ndfloor'] = df_train_add['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_train_add['hasgarage'] = df_train_add['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_train_add['hasbsmt'] = df_train_add['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_train_add['hasfireplace'] = df_train_add['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df_test_add['haspool'] = df_test_add['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_test_add['has2ndfloor'] = df_test_add['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_test_add['hasgarage'] = df_test_add['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_test_add['hasbsmt'] = df_test_add['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_test_add['hasfireplace'] = df_test_add['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[191]:


df_train[df_train["SalePrice"] > 600000 ] #Discovering the outliers


# In[192]:


categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)
print(categorical_values_train)


# In[196]:


for i in categorical_values_train:
    feature_set = set(df_train[i])
    for j in feature_set:
        feature_list = list(feature_set)
        df_train.loc[df_train[i] == j, i] = feature_list.index(j)
        df_train_add.loc[df_train[i] == j, i] = feature_list.index(j)

for i in categorical_values_test:
    feature_set2 = set(df_test[i])
    for j in feature_set2:
        feature_list2 = list(feature_set2)
        df_test.loc[df_test[i] == j, i] = feature_list2.index(j)
        df_test_add.loc[df_test[i] == j, i] = feature_list2.index(j)


# In[197]:


df_train_add.head()


# In[198]:


df_test_add.head()


# In[199]:


df_train_add.dtypes


# In[200]:


#df_train_add = df_train_add.astype('int64')
#df_test_add = df_test_add.astype('int64')


# In[201]:


cor_matrix = df_train_add.corr()
f, ax = plt.subplots(figsize=(30, 19))
sns.set(font_scale=2)
sns.heatmap(cor_matrix,square = True)


# In[202]:


features = cor_matrix.SalePrice.sort_values(ascending = False)[1:10].index


# In[203]:


cor_matrix.SalePrice.sort_values(ascending = False)


# ### 6. ML Models

# In[204]:


#Importing all the librairies we'll need

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold


# In[205]:


X_train = df_train_add.drop(["SalePrice","log_sales_price"], axis=1)
y_train = df_train_add["log_sales_price"]


# In[206]:


from sklearn.model_selection import train_test_split
X_training, X_valid, y_training, y_valid = train_test_split(X_train,y_train,test_size = 0.2, random_state=0)


# ### Linear Regression Model

# ### Adding GridSearchCV Function

# In[207]:


linreg = LinearRegression()
parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}
grid_linreg = GridSearchCV(linreg, parameters_lin, verbose = 1, scoring = "r2")
grid_linreg.fit(X_training,y_training)

print("Best Score: " + str(grid_linreg.best_score_))


# In[208]:


linreg = grid_linreg.best_estimator_
linreg.fit(X_training, y_training)
lin_pred = linreg.predict(X_valid)
r2_lin = r2_score(y_valid, lin_pred)
rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))
print("R^2 Score: " + str(r2_lin))
print("RMSE Score: " + str(rmse_lin))


# In[209]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_valid, lin_pred))
print('MSE:', metrics.mean_squared_error(y_valid, lin_pred))


# In[210]:


scores_lin = cross_val_score(linreg, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lin)))


# ### Ridge Model

# In[211]:


ridge = Ridge()
parameters_ridge = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False], "solver" : ["auto"]}
grid_ridge = GridSearchCV(ridge, parameters_ridge, verbose=1, scoring="r2")
grid_ridge.fit(X_training, y_training)

print("Best Ridge Model: " + str(grid_ridge.best_estimator_))
print("Best Score: " + str(grid_ridge.best_score_))


# In[212]:



ridge = grid_ridge.best_estimator_
ridge.fit(X_training, y_training)
ridge_pred = ridge.predict(X_valid)
r2_ridge = r2_score(y_valid, ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_valid, ridge_pred))
print("R^2 Score: " + str(r2_ridge))
print("RMSE Score: " + str(rmse_ridge))


# In[213]:


scores_ridge = cross_val_score(ridge, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_ridge)))


# ### Xgboost

# In[ ]:





# In[229]:


from xgboost import XGBRegressor

xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=20000,
                       max_depth=3, 
                       min_child_weight=0,
                       gamma=0, 
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', 
                       nthread=-1,
                       scale_pos_weight=1, 
                       seed=27,
                       reg_alpha=0.006)


# In[230]:


xgb = xgboost.fit(X_training.select_dtypes([np.number]), y_training)


# In[232]:


xgb_pred = xgb.predict(X_valid.select_dtypes([np.number]))
r2_xgb = r2_score(y_valid, xgb_pred)
rmse_xgb = np.sqrt(mean_squared_error(y_valid, xgb_pred))
print("R^2 Score: " + str(r2_xgb))
print("RMSE Score: " + str(rmse_xgb))


# In[235]:





# In[236]:


submission_predictions = np.exp(xgb.predict(df_test_add.select_dtypes([np.number])))
print(submission_predictions)


# In[239]:


res=pd.DataFrame(columns = ['Id', 'SalePrice'])
res['Id'] = df_test.index + 1461
res['SalePrice'] = submission_predictions
res.to_csv('submission1.csv',index=False)


# In[238]:





# In[ ]:




