import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics
# import sklearn.metrics.r2_scorer
from numpy.distutils.command.install_clib import install_clib
from numpy.random import RandomState
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# load data to data frame
df = pd.read_csv('House_Data.csv')

# split dataframe data, output and expand output dims
X = df.iloc[:, 1:77]
y = df['SalePrice']
y = np.expand_dims(y, axis=1)

# delete this col BsmtHalfBath most value equal 0
# delete this col Fence and MiscFeature most val is NA
del X['BsmtHalfBath']
del X['Fence']
del X['MiscFeature']
del X['PoolQC']
del X['Utilities']
# apply preprocessing to all data then split it

# 1st split numerical col and cat
# split data to numerical and categories

# split data train and test data
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, shuffle=True)


numerical_train = X_train.select_dtypes(exclude='object')
cat_train = X_train.select_dtypes(include='object')
# class sklearn.impute.SimpleImputer(*, missing_values=nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)[source]
# to show how Impute work can replace missing value with mean, median, most_frequent, constant
imputer_numerical_ = SimpleImputer(missing_values=np.nan, strategy='mean').fit(numerical_train)

#imputer_numerical = imputer.transform(numerical)
imputer_numerical = pd.DataFrame(imputer_numerical_.transform(numerical_train), columns=numerical_train.columns)
# apply feature selection in numerical cols
corr = imputer_numerical.corr()
#Top 50% Correlation training features with the SalePrice
top_feature_num = corr.index[abs(corr['SalePrice'] > 0.4)]

top_feature = top_feature_num.delete(-1)
imputer_numerical = imputer_numerical[top_feature_num]

plt.subplots(figsize=(12, 8))
top_corr = X[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

cat_train['SalePrice'] = X['SalePrice']
# apply label encode to cat convert from label to number
for c in cat_train:
    lbl = LabelEncoder()
    lbl.fit(list(X[c].values))
    cat_train[c] = lbl.transform(list(cat_train[c].values))
# fill nan values in cat to most_frequent value
imputer_cat_ = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit(cat_train)
#imputer_numerical = imputer.transform(numerical)
imputer_cat = pd.DataFrame(imputer_cat_.transform(cat_train), columns=cat_train.columns)



# concatunate Sale col to calculate correlation to cat
# imputer_cat['SalePrice'] = X['SalePrice']

corr = imputer_cat.corr()
#Top 70% Correlation training features with the SalePrice
top_feature_cat = corr.index[abs(corr['SalePrice']>0.4)]

top_feature = top_feature_cat.delete(-1)
imputer_cat = imputer_cat[top_feature_cat]

# con the data
con_train = pd.concat([imputer_numerical, imputer_cat], axis=1)
# delete the output column
del con_train['SalePrice']

#imputer_numerical = pd.DataFrame(imputer_numerical.transform(numerical_train), columns=numerical_train.columns)

numerical_test = X_test.select_dtypes(exclude='object')
cat_test = X_test.select_dtypes(include='object')

imputer_num_test = pd.DataFrame(imputer_numerical_.transform(numerical_test), columns=numerical_test.columns)
# apply feature selection in numerical cols
corr = imputer_num_test.corr()
#Top 50% Correlation training features with the SalePrice
top_feature_test = top_feature_num.delete(-1)
imputer_num_test = imputer_num_test[top_feature_num]
cat_test['SalePrice'] = X['SalePrice']
for c in cat_test:
    lbl = LabelEncoder()
    lbl.fit(list(X[c].values))
    cat_test[c] = lbl.transform(list(cat_test[c].values))


imputer_cat_test = pd.DataFrame(imputer_cat_.transform(cat_test), columns=cat_test.columns)
top_feature_test = top_feature_cat.delete(-1)
imputer_cat_test = imputer_cat_test[top_feature_test]

con_test=pd.concat([imputer_num_test, imputer_cat_test], axis=1)
del con_test['SalePrice']
# apply first model

cls = linear_model.LinearRegression()
cls.fit(con_train,y_train)
prediction= cls.predict(con_train)

print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_train), prediction))
print('accuracy', r2_score(y_train, prediction)*100)

prediction= cls.predict(con_test)

print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
print('accuracy',r2_score(y_test,prediction )*100)
#sklearn.metrics.r2_score(y, prediction, *, sample_weight=None, multioutput='uniform_average')
