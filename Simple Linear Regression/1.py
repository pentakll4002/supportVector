import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:\DS\Simple Linear Regression\height-weight.csv')
print(df)

df.head()
df.tail()

### Correllation Data Frame df['Weight'] and df['Height']
plt.scatter(df['Weight'], df['Height'])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()

### Devide our dataset into independent and dependent features
X = df[['Weight']] ### Independent feature
y = df[['Height']] ### Dependent feature
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape), (X_test.shape), (y_train.shape), print(y_test.shape)

### Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plt.scatter(X_train, y_train)
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()

### Train Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("The slope or coefficient of weight is: ", regressor.coef_)
print("Intercept: ", regressor.intercept_)
plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train), 'r')
plt.show()

#    Prediction of train data
###  Predicted height output = intercept + coef_(Weights)
#### y_pred_train = 157.5 + 17.03(X_train)
#### prediction of test data
#### predicted height output = intercept + coef_(Weights)
#### y_pred_test = 157.5 + 17.03(X_test)

y_pred_test = regressor.predict(X_test)
print(y_pred_test)
print(y_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, regressor.predict(X_test), 'r')
plt.show()

# Performance Metrics
#### 1. MSE, MAE, RMSE
#### 2. R-Square, R-Adjust-Squared
### MSE, MAE, RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
print('MAE: ', mae)
print('MSE: ', mse)
print('RMSE: ', rmse)


# R square
# Formula
### R² = 1 - SSR/SST
#### R² = coefficient of determination
#### SSR = sum of squares of residuals
#### SST = total sum of squares
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred_test)
print('R2: ', score)


## Adjusted R² = 1 - [(1 - R²)(n - 1)] / (n - k - 1)
## where:
####R²: The R² of the model
####n: The number of observations
####k: The number of predictor variables
r2_adjust = 1 - (1 - score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
print("R2 Adjust Squared: ", r2_adjust)


## New data point weight is 90
scaled_weight = scaler.transform([[90]])
print(scaled_weight[0])
print("The height prediction for weight 90 kg is: ", regressor.predict([scaled_weight[0]]))

## Assumption
## Plot a scatter plot for the prediction
plt.scatter(y_test, y_pred_test)
plt.show()

## Residuals
residuals = y_test - y_pred_test
print(residuals)

### Plot this residuals
sns.histplot(residuals, kde=True)

## Scatter plot with respect to predictions and residuals
## Uniform distribution
plt.scatter(y_pred_test, residuals)