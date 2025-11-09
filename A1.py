//MAINLY SKKIP

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("uber.csv")
print(df.head())

df = df.dropna()
df = df[df['fare_amount'] > 0]
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

plt.figure(figsize=(5,3))
sns.boxplot(df['fare_amount'])
plt.title("Outlier Detection")
plt.show()

df = df[df['fare_amount'] < 100]

features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
X = df[features]
y = df['fare_amount']

plt.figure(figsize=(6,4))
sns.heatmap(df[features + ['fare_amount']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(name, "R2:", round(r2, 3), "RMSE:", round(rmse, 3))

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest Regression")
