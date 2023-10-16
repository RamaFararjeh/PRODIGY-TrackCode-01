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

#  NOTE : BedroomAbvGr => represents the number of bedrooms above ground in a house.   

# if 'BedroomAbvGr' in df:
#     print('True')
# else:
#     print('False')

# print(df['BedroomAbvGr'].value_counts())

features=['SalePrice','LotArea','BedroomAbvGr','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']

for i in df:
    if i not in features:
        df.drop(i,axis=1,inplace=True)

print(df)
print(df.shape)
print(df.head())
print(df.info())
print(df.describe().T)


# check on missing value for all features.
for i in df:
    # print(i)
    print(f'{i} : {df[i].isnull().sum()}')

# There is no Missing Values.
#----------------------------------------------------------------------------------------

#  NOTE : We don't need to Separating categorical and numarical columns CUZ all features int64 data type.

# Histogram for Distribution.
def hist_plot():
    for i in df:
        sns.histplot(df[i],bins=20,kde=True)
        plt.hist(df[i],bins=20,edgecolor='k',color='darkblue')
        plt.xlabel(i)
        plt.ylabel('Frequancy')
        plt.title(f'Histogram of {i}')
        plt.show()

#----------------------------------------------------------------------------------------

# Box plot for Outliers in num_data.
def boxplot():
    for i in df:
        sns.boxplot(df[i])
        plt.xlabel(i)
        plt.ylabel('Frequancy')
        plt.title(f'Box plot of {i}')
        plt.show()


#----------------------------------------------------------------------------------------

# Heatmap:- => creates a correlation matrix heatmap to explore relationships between all pairs of numeric variables.

corr_matrix=df.corr()
plt.figure(figsize=(8,6))
plt.title('Correlation Matrix')
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f",center=0)
print(f'corr_matrix : \n {corr_matrix}')
# plt.show()


# Creat a correlations between individual numeric features and the target variable 'SalePrice.' only  

correlation = df.corrwith(df['SalePrice'])
correlation = correlation.sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation.to_frame(), annot=True, cmap='coolwarm')
plt.title('Correlation with SalePrice') # WITH TARGET FEATURES ONLY.
# plt.show()

#----------------------------------------------------------------------------------------


# Scatter

# Features with target

def scatterr():
    for i in df:
                
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


# print(df['LotArea'].value_counts())
# print(df['FullBath'].value_counts())
# print(df['BedroomAbvGr'].value_counts())




#----------------------------------------------------------------------------------------

# hist_plot()
# boxplot()
# scatterr()

#----------------------------------------------------------------------------------------

# Split Train data for Validation.
x=df.drop('SalePrice',axis=1)
y=df['SalePrice']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(f'x_train {x_train}')
print(f'y_train {y_train}')
#----------------------------------------------------------------------------------------

# Scaling For Numarical Features 

num_col=x_train.select_dtypes(include=['int64']).columns # kant hoon almoshkelah

scaler=MinMaxScaler()

# Fit
scaler.fit(x_train[num_col])

# Transform
x_train[num_col]=scaler.transform(x_train[num_col])
x_test[num_col]=scaler.transform(x_test[num_col])

print(x_train)

#----------------------------------------------------------------------------------------


# Random Forest Model
# n_estimators => determines the number of decision trees ..
# it means that you are creating a Random Forest ensemble consisting of 100 decision trees.

# Initialize the Random Forest Regressor
rf_model=RandomForestRegressor(n_estimators=500,random_state=42)

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


# Insight:
# --------
# Insights:
# ----------
# 1. LotArea: The distribution of lot areas is positively skewed, indicating that there may be some properties with very large lot areas, possibly outliers.
# 2. BedroomAbvGr: The number of bedrooms above ground is positively correlated with SalePrice. This suggests that houses with more bedrooms tend to have higher sale prices.
# 3. FullBath and HalfBath: Both FullBath and HalfBath variables have relatively high positive correlations with SalePrice. This indicates that the number of bathrooms, both full and half, is an important factor in predicting sale prices.

# 5. Scaling: Min-Max scaling was applied to ensure that numerical features are on a consistent scale for modeling, preventing some features from dominating the learning process due to their larger scale.
# 6. Random Forest Model: A Random Forest model with 500 decision trees was trained to predict house prices. The ensemble nature of Random Forests can lead to more accurate predictions.
# 7. Model Evaluation: The model was evaluated using various metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²). These metrics provide a comprehensive assessment of the model's performance.


#  Special thanks to Prodigy for the opportunity to work on this project and gain valuable skills in data science and machine learning.
