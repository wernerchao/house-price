import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Uncomment below 2 lines to make plots
# import matplotlib.pyplot as plt
# import pylab


train = pd.read_csv("kc_house_train_data.csv")
test = pd.read_csv("kc_house_test_data.csv")



# Step (1) making new features by:
train["bedrooms_squared"] = train["bedrooms"] * train["bedrooms"]
train["bed_bathrooms"] = train["bedrooms"] * train["bathrooms"]
train["log_sqft_living"] = np.log(train["sqft_living"])
train["lat_plus_long"] = train["lat"] + train["long"]



# Step (2) making new features for TESTING data
test["bedrooms_squared"] = test["bedrooms"] * test["bedrooms"]
test["bed_bathrooms"] = test["bedrooms"] * test["bathrooms"]
test["log_sqft_living"] = np.log(test["sqft_living"])
test["lat_plus_long"] = test["lat"] + test["long"]



# Step (2.5) Make plots to inspect relationships between features and response.
# Uncomment below to see all plots.
# for col in train:
#     if col == "date" or col == "id" or col == "price":
#         continue
#     elif col == "bathrooms":
#     # else:
#         train.plot(col, "price", kind='scatter')
#         plt.show()



# Step (3) taking the mean of the new test features
mean_bedrooms_squared = np.mean(test["bedrooms_squared"])
mean_bed_bathrooms = np.mean(test["bed_bathrooms"])
mean_log_sqft_living = np.mean(test["log_sqft_living"])
mean_lat_plus_long = np.mean(test["lat_plus_long"])


# Step (4) aggregating feature columns for modelling
# Model_1, train & test
model_1_train = train[list(train.columns[3:6]) + list(train.columns[17:19])]
model_1_test = test[list(test.columns[3:6]) + list(test.columns[17:19])]



# Step (4.5) polynomial feature transformation
poly = PolynomialFeatures(degree=2)
poly_model_1_train = poly.fit_transform(model_1_train)
print poly_model_1_train.shape
poly_model_1_test = poly.fit_transform(model_1_test)
print poly_model_1_test.shape


# Step (5) fitting and predicting
lr1 = linear_model.LinearRegression()
lr1.fit(model_1_train, train['price'])
print lr1.coef_


lr1_poly = linear_model.LinearRegression()
lr1_poly.fit(poly_model_1_train, train['price'])
print lr1_poly.coef_


pred1 = lr1.predict(model_1_test)
mse1 = mean_squared_error(test['price'], pred1)
pred1_poly = lr1_poly.predict(poly_model_1_test)
mse1_poly = mean_squared_error(test['price'], pred1_poly)


print "Printing MSE of both models: "
print mse1, mse1_poly



score1 = lr1.score(model_1_test, test['price'])
score1_poly = lr1_poly.score(poly_model_1_test, test['price'])
print "Printing SCORE of both models:"
print score1, score1_poly
