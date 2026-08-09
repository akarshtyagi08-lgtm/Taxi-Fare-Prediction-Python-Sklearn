# Import required Modules
import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reading Data
path = kagglehub.dataset_download("hrishikeshsuresh/taxi-fare-data-2023")
file_path = os.path.join(path, "Taxi_Trip_Data.csv")
data = pd.read_csv(file_path)

# Encode Data using OneHot Encoding
data = pd.get_dummies(data, columns=['payment_type'], drop_first=True)

# Initilize Model
model = LinearRegression()

# Setup Input and Output
X = data.drop(columns=['fare_amount'])
y = data['fare_amount']

# Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on training data
model.fit(X_train, y_train)

# Test the model on testing data
y_pred = model.predict(X_test)

# Get Overall Report
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the Report
print(f"MAE: {round(mae, 3)}\nMSE: {round(mse, 3)}\nRMSE: {round(rmse, 3)}\nR^2 Score: {round(r2, 5)}")
