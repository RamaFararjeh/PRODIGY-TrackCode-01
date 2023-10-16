import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import f_oneway
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error ,r2_score



df=pd.read_csv(r'C:\Users\admin\Development\Classification\Machine-Learning\Data-Analysis\House_Price\train.csv')
print(len(df.columns)) # number of features

df.drop(['Id','Alley','MiscFeature','Fence','PoolQC','GarageFinish','GarageCond','TotRmsAbvGrd','TotalBsmtSF','GarageYrBlt',
            'FireplaceQu','LandSlope','Neighborhood'],axis=1,inplace=True)

print(df.columns)
print(df.shape)
print(df.head())
print(df.info())
print(df.describe().T)
print(df['LotFrontage'].value_counts())

# print(f'Number of duplicated data : \n {df.duplicated().sum()}') # There is no duplicate data.



# check on missing value for all features.
for i in df:
    # print(i)
    print(f'{i} : {df[i].isnull().sum()}')

#----------------------------------------------------------------------------------------

# Separating categorical and numarical columns
cat_data=df.select_dtypes(include='object').columns
num_data=df.select_dtypes(include=['int64', 'float64']).columns

print(cat_data) # as list
# print(df[cat_data])

# print(num_data) # as list
print('-------------------------------------------------------------')
print(f'Descirbe Category data : \n{df[cat_data].describe().T}')
print('-------------------------------------------------------------')
print(f'Describe Numarical data : \n{df[num_data].describe().T}')
#----------------------------------------------------------------------------------------

# Histogram for Distribution.
def hist_plot():
    for i in num_data:
        sns.histplot(df[i],bins=20,kde=True)
        plt.hist(df[i],bins=20,edgecolor='k',color='darkblue')
        plt.xlabel(i)
        plt.ylabel('Frequancy')
        plt.title(f'Histogram of {i}')
        plt.show()

#----------------------------------------------------------------------------------------

# Box plot for Outliers in num_data.
def boxplot():
    for i in num_data:
        sns.boxplot(df[i])
        plt.xlabel(i)
        plt.ylabel('Frequancy')
        plt.title(f'Box plot of {i}')
        plt.show()

#----------------------------------------------------------------------------------------

# Heatmap:- => creates a correlation matrix heatmap to explore relationships between all pairs of numeric variables.

corr_matrix=df[num_data].corr()
plt.figure(figsize=(12,10))
plt.title('Correlation Matrix')
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f",center=0)
print(f'corr_matrix : \n {corr_matrix}')
# plt.show()

# Creat a correlations between individual numeric features and the target variable 'SalePrice.' only  

correlation = df[num_data].corrwith(df['SalePrice'])
correlation = correlation.sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation.to_frame(), annot=True, cmap='coolwarm')
plt.title('Correlation with SalePrice') # WITH TARGET FEATURES ONLY.
# plt.show()

# NOTE:
#correlation.to_frame()=>[ method is used to convert a Pandas Series object into a Pandas DataFrame.]
#  you are converting this one-dimensional Series into a two-dimensional DataFrame
#  where one column of the DataFrame will contain the data from the original Series, 
# and the DataFrame will have a column label 
#----------------------------------------------------------------------------------------


# Stacked bar chart for categorical data only.
def stacked_bar_chart():
    for i in df[cat_data]:
        cross_tab = pd.crosstab(df[i],df[i])
        cross_tab.plot(kind='bar', stacked=True)
        plt.xlabel(i)
        plt.ylabel('Frequency')
        plt.title('Stacked Bar Chart for Categorical Data')
        plt.legend(title=i)
        plt.show()

# stacked_bar_chart()


#----------------------------------------------------------------------------------------

def cat_with_target():
    for i in cat_data:
        plt.figure(figsize=(8,6))
        sns.barplot(x=i, y=df['SalePrice'], data=df)
        plt.xlabel(i)
        plt.ylabel('SalePrice')
        plt.title(f'cat_data - Bar Plot of {i} vs. {df["SalePrice"]}')
        plt.show()

#----------------------------------------------------------------------------------------

# Scatter

# Features with target

def scatterr():
    for i in num_data:
                
        plt.figure(figsize=(4,6))
        plt.scatter(df[i],df['SalePrice'],c='yellow',marker='s',label='Data Points')

        # Marker Value :
        # 'o' for circular.
        # 's' for square.
        # '^' for triangle.

        # label  => legend

        plt.xlabel(i)
        plt.ylabel('SalePrice')
        plt.title(f'Scatter Plot {i} vs SalePrice ')
        plt.legend() # مفتاح الخريطه
        plt.show()


#----------------------------------------------------------------------------------------

# boxplot()
# hist_plot()

# scatterr() #laaaqqq
# cat_with_target()

#----------------------------------------------------------------------------------------

# Interactive plot for all features with target .

for i in df.columns:
    if i != 'SalePrice':  # Exclude the target variable
        # Create an interactive bar plot for each feature
        fig = px.bar(df, x=i, y='SalePrice', color=i,
                title=f'Sale Price by {i}',
                labels={i: i, 'SalePrice': 'Sale Price'},
                text='SalePrice')

        # Show the interactive plot
        # fig.show()

#----------------------------------------------------------------------------------------

# Convert all categorical col to numerical col using Label Encoder.

label_encoder = LabelEncoder() # instance of LabelEncoder class
columns_to_encode = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
                    'LotConfig', 'Condition1', 'Condition2',
                    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                    'Functional', 'GarageType', 'GarageQual', 'PavedDrive',
                    'SaleType', 'SaleCondition']
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

print(f'Encoded Data : \n {df}')
# print(df.info()) # in this line we check on data type for all features in dataset.
# print('rama')
# print(df[cat_data])
#----------------------------------------------------------------------------------------
# Split Train data for Validation.
x=df.drop('SalePrice',axis=1)
y=df['SalePrice']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# print(y_train)
#----------------------------------------------------------------------------------------

# Handling With Missing Value.


# Create a SimpleImputer for filling missing values with the median
imputer = SimpleImputer(strategy='median')

columns_to_impute = ['LotFrontage', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond',
                    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
                    'GarageType', 'GarageQual']
# Fit the imputer on the 'Age' column of the training data
imputer.fit(x_train[columns_to_impute])

# Transform the 'Age' column of both the training and test data using the fitted imputer
x_train[columns_to_impute] = imputer.transform(x_train[columns_to_impute])
x_test[columns_to_impute] = imputer.transform(x_test[columns_to_impute])

print(x_train)
print(x_train.isna().sum())

#----------------------------------------------------------------------------------------

# Scaling For Numarical Features 

num_col=x_train.select_dtypes(include=['float64']).columns # kant hoon almoshkelah

scaler=MinMaxScaler()

# Fit
scaler.fit(x_train[num_col])

# Transform
x_train[num_col]=scaler.transform(x_train[num_col])
x_test[num_col]=scaler.transform(x_test[num_col])


#----------------------------------------------------------------------------------------

# Random Forest Model
# n_estimators => determines the number of decision trees ..
# it means that you are creating a Random Forest ensemble consisting of 100 decision trees.

# Initialize the Random Forest Regressor
rf_model=RandomForestRegressor(n_estimators=100,random_state=42)

# Train the model
rf_model.fit(x_train, y_train)  


# Make predictions on the test data
y_pred = rf_model.predict(x_test)

# Calculate evaluation metrics
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2=r2_score(y_test,y_pred)

print(f'y_predict {y_pred}')
print(f'mean_absolute_error : {MAE}')
print(f'mean_squared_error  : {MSE}')
print(f'Root mean_squared_error : {RMSE}')
print(f'R-squared  : {R2}')
print('----------------------------------------------------------------------------------------')








# about Data
# -------------------
# SalePrice => target features that you're trying to predict.
# MSSubClass: The building class
# MSZoning : most frequent zone by populous is RL [ Residential low Density ]
# LotFrontage : ---
# LotArea: Lot size in square feet حجم قطعه الارضيه 
# Street: Type of road access
# LotShape: General shape of property
# 



# Insight:
# --------
# MSSubClass => The most Frequent class.[20 : 536]
# -----------------------------------------------------------------------------
# [ LandContour ] => feature can be important in a house price dataset ... 
#  CUZ it provides information about the topography of the land
# in LandContour Features I concluded that the flat land [ lvl ] are inhabited by more ppl and have higher price.
# also about [ Hls ] : House on the hills offer stunning views , although construction is diffcult , & the price ..
# depends on the stunning views.
# also in [ low ] : low lying area => Susceptible to drainage issues or flooding which can effect their usability... 
# and have lower average sale price due to potential concerns related to flooding.
# -----------------------------------------------------------------------------
# [ Utilities  ]
# 1) "NoSeWa" utilities may be less attractive ( 1 )
# 2) "AllPub" utilities are typically more desirable and practical for homeowners. ( 1459 )
# -----------------------------------------------------------------------------
# [ LotConfig ]
# Corner : have higher average sale prices due to their unique characteristics - more accessible parking.
# CulDSac :  have higher average sale prices due providing a quieter and safer environment, especially for families with children.
# Inside : Inside lots may have average sale prices, but they are often less expensive than corner or cul-de-sac .
# FR2 : Frontage on 2 sides of property.
# FR3 : Frontage on 3 sides of property.
# -----------------------------------------------------------------------------
# [ LandSlope ] => I will drop this column because I can get the same result from LandContour Features...
# have GTL => Gentle slope - easier to bulid - have higher avg sale price.
# also Mod => Moderate slope - have some challenge for constuction but Generall manageable - price depending on the degree of slope.
# also Sev => Sever Slpoe - Severe slopes can be challenging for construction - have drainage issues - Have lower avg sale price .
# -----------------------------------------------------------------------------
# [ Neighborhood ]
# "NAmes" is the most common neighborhood in the dataset, with 225 occurrences.
# "CollgCr" is the second most common neighborhood, with 150 occurrences.
# "OldTown" is the third most common neighborhood, with 113 occurrences.
# -----------------------------------------------------------------------------
# [ Street ] Type of road access to property 
# - Grvl (Gravel) => less expensive to build and maintain compared to paved roads. حصى
# - Pave (Paved) => more expensive to construct and maintain. رصيف
# -----------------------------------------------------------------------------
# [ BldgType ] Type of dwelling
# 1Fam => Single-family Detached
# 2FmCon => Two-family Conversion; originally built as one-family dwelling
# Duplx => Duplex
# TwnhsE => Townhouse End Unit
# TwnhsI => Townhouse Inside Unit
# 
# -----------------------------------------------------------------------------
# [ LotFrontage ] provides information about how wide the front of the property's land is along the street or road.
# 
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


