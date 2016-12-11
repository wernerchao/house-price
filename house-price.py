import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error


train = pd.read_csv("kc_house_train_data.csv")
test = pd.read_csv("kc_house_test_data.csv")



# Step (1) making new features by:
# 1) squaring "bedrooms", 2) making "bedrooms" * "bathrooms", 3) logging "sqft_living", 
# 4) adding "lat + long", for TRAINING data
train["bedrooms_squared"] = train["bedrooms"] * train["bedrooms"]
train["bed_bathrooms"] = train["bedrooms"] * train["bathrooms"]
train["log_sqft_living"] = np.log(train["sqft_living"])
train["lat_plus_long"] = train["lat"] + train["long"]



# Step (2) making new features for TESTING data
test["bedrooms_squared"] = test["bedrooms"] * test["bedrooms"]
test["bed_bathrooms"] = test["bedrooms"] * test["bathrooms"]
test["log_sqft_living"] = np.log(test["sqft_living"])
test["lat_plus_long"] = test["lat"] + test["long"]

# Printing values to make sure
# print train.shape
# print test.shape
print train.head(5)



# Step (3) taking the mean of the new test features
mean_bedrooms_squared = np.mean(test["bedrooms_squared"])
mean_bed_bathrooms = np.mean(test["bed_bathrooms"])
mean_log_sqft_living = np.mean(test["log_sqft_living"])
mean_lat_plus_long = np.mean(test["lat_plus_long"])

# Printing out values to make sure
# print mean_bedrooms_squared, mean_bed_bathrooms, mean_log_sqft_living, mean_lat_plus_long
# print train["sqft_living"].shape, train["bedrooms"].shape, train["bathrooms"].shape, train["lat"].shape, train["long"].shape
# print len(train.index)


# Step (4) making features for modelling
# Model_1, train & test
model_1_train = train[list(train.columns[3:6]) + list(train.columns[17:19])]
model_1_test = test[list(test.columns[3:6]) + list(test.columns[17:19])]
# print model_1_train.head(5)

# Model_2, train & test
model_2_train = train[list(train.columns[3:6]) + list(train.columns[17:19]) + list(train.columns[22:23])]
model_2_test = test[list(test.columns[3:6]) + list(test.columns[17:19]) + list(test.columns[22:23])]
# print model_2_train.head(5)

# Model_3, train & test
model_3_train = train[list(train.columns[3:6]) + list(train.columns[17:19]) + list(train.columns[21:25])]
model_3_test = test[list(test.columns[3:6]) + list(test.columns[17:19]) + list(test.columns[21:25])]
# print model_3_train.head(5)


# Step (5) fitting and predicting
lr = linear_model.LinearRegression()
lr.fit(model_1_train, train['price'])
print lr.coef_
# [ 'bedrooms'        'bathrooms'     'sqft_living'    'lat'            'long'          ]
# [ -5.95865332e+04   1.57067421e+04   3.12258646e+02   6.58619264e+05   -3.09374351e+05]

pred = lr.predict(model_1_train)
mse = mean_squared_error(train['price'], pred)
# print pred.shape
print mse

