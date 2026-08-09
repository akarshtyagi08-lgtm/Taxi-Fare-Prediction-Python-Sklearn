# Taxi-Fare-Prediction-Python-Sklearn
This is a ML (Machine Learning) Project, this model predicts taxi fare based on 4 parameters, the model was trained on 6,760,872 rows and tested on 1,690,218 rows, the model got ~97% Accuracy on testing data. LinearRegression was used.

# Overall Report
```
MAE: 0.483                                            MSE: 0.746                                            RMSE: 0.864                                           R^2 Score: 0.97389
```

# Modules Used in the project
1. numpy
2. pandas
3. matplotlib
4. seaborn
5. sklearn (Scikit-learn)

NOTE:- You don't need **matplotlib** and **seaborn** as they were used for visualization of data, which is already done.

# More Details
The dataset used in this project is imported from **Kaggle**. The Heatmap is also uploaded, you can use and train the model on More bigger dataset.

# Process
I took the dataset and cleaned all the Null values (NaN or None values), then I took the data and performed OneHot encoding on it using pandas.
```
# Encode Data using OneHot Encoding
data = pd.get_dummies(data, columns=['payment_type'], drop_first=True)
```
Then the data was splited into Training and Testing data. I took the model and trained is on LinearRegression, after testing the report was:
```
MAE: 0.483                                            MSE: 0.746                                            RMSE: 0.864                                           R^2 Score: 0.97389
```
