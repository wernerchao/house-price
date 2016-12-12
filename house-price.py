import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Uncomment below 2 lines to make plots
import matplotlib.pyplot as plt
import pylab


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
# print train.head(5)



# Step (2.5) Make plots to inspect relationships between features and response.
# Uncomment below to see all plots.
for col in train:
    if col == "date" or col == "id" or col == "price":
        continue
    # elif col == "bathrooms":
    else:
        train.plot(col, "price", kind='scatter')
        plt.show()



# Step (3) taking the mean of the new test features
mean_bedrooms_squared = np.mean(test["bedrooms_squared"])
mean_bed_bathrooms = np.mean(test["bed_bathrooms"])
mean_log_sqft_living = np.mean(test["log_sqft_living"])
mean_lat_plus_long = np.mean(test["lat_plus_long"])
# Printing out values to make sure
print "Mean of new features on TEST data:"
print mean_bedrooms_squared, mean_bed_bathrooms, mean_log_sqft_living, mean_lat_plus_long
# print train["sqft_living"].shape, train["bedrooms"].shape, train["bathrooms"].shape, train["lat"].shape, train["long"].shape
# print len(train.index)


# Step (4) aggregating feature columns for modelling
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
lr1 = linear_model.LinearRegression()
lr1.fit(model_1_train, train['price'])
print model_2_train.head(0) # Check which column is which feature
print lr1.coef_
# [ 'bedrooms'        'bathrooms'     'sqft_living'    'lat'            'long'          ]
# [ -5.95865332e+04   1.57067421e+04   3.12258646e+02   6.58619264e+05   -3.09374351e+05]

lr2 = linear_model.LinearRegression()
lr2.fit(model_2_train, train['price'])
print model_2_train.head(0) # Check which column is which feature
print lr2.coef_

lr3 = linear_model.LinearRegression()
lr3.fit(model_3_train, train['price'])

pred1 = lr1.predict(model_1_test)
mse1 = mean_squared_error(test['price'], pred1)
pred2 = lr2.predict(model_2_test)
mse2 = mean_squared_error(test['price'], pred2)
pred3 = lr3.predict(model_3_test)
mse3 = mean_squared_error(test['price'], pred3)
print "Printing MSE of all 3 models: "
print mse1, mse2, mse3



score1 = lr1.score(model_1_test, test['price'])
score2 = lr2.score(model_2_test, test['price'])
score3 = lr3.score(model_3_test, test['price'])
print "Printing SCORE of all 3 models:"
print score1, score2, score3




### ---------------------------------------------------------------------------------------------------
### Below code plot the prediction with the training data of 1 feature
# Reshaping the train feature so it can be fitted
# temp_log_sqft_living = train['log_sqft_living'].reshape(len(train.index),1)

# Fit and predict
# lr = linear_model.LinearRegression()
# lr.fit(temp, train['price'])
# pred = lr.predict(temp_log_sqft_living)

# Reshape the test feature so it can be scored
# test_log_sqft_living = test['log_sqft_living'].reshape(len(test.index),1)

# Check the score. 1 is highest
# score = lr.score(test_log_sqft_living, test['price'])
# print train['log_sqft_living'].shape, pred.shape

# Make some plots to see the prediction
# plt.plot(train["log_sqft_living"], train["price"], 'x')
# plt.plot(train["log_sqft_living"], pred, '.')
# plt.show()