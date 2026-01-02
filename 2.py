
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("StudentPerformance.csv")

print(df.head(10))
print(df.tail(10))
print(df.info())
print(df.describe)
print(df.isna().sum())
print(df.duplicated().sum())

df = pd.get_dummies(df, columns=["Extracurricular Activities"], drop_first=True)

x = df.drop(columns=["Performance Index"])
y = df["Performance Index"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)
print(x_scaled_df)

x_train, x_test, y_train, y_test = train_test_split(x_scaled_df, y, train_size=0.8, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("prediction:", y_pred)
print("actual:", y_test.values)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("mse:", mse)
print("r2:", r2)

coef_series = pd.Series(model.coef_.ravel(), index=x.columns).sort_values(key=abs, ascending=False)
print(coef_series)
coef_series.plot(kind='bar', figsize=(8,6), title="Feature Importance")
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(6,4))                   
plt.scatter(y_pred, residuals, alpha=0.7)    
plt.axhline(y=0, linestyle='--', color="r")  
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()


