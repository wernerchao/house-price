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
model_2_train = train[list(train.columns[3:6]) + list(train.columns[17:19]) + list(train.columns[22:23])]
model_2_test = test[list(test.columns[3:6]) + list(test.columns[17:19]) + list(test.columns[22:23])]
model_3_train = train[list(train.columns[3:6]) + list(train.columns[17:19]) + list(train.columns[21:25])]
model_3_test = test[list(test.columns[3:6]) + list(test.columns[17:19]) + list(test.columns[21:25])]


# Step (4.5) polynomial feature transformation
poly = PolynomialFeatures(degree=2)
poly_model_1_train = poly.fit_transform(model_1_train)
poly_model_1_test = poly.fit_transform(model_1_test)
poly_model_2_train = poly.fit_transform(model_2_train)
poly_model_2_test = poly.fit_transform(model_2_test)
poly_model_3_train = poly.fit_transform(model_3_train)
poly_model_3_test = poly.fit_transform(model_3_test)
print poly_model_1_test.shape


# Step (5) fitting and predicting
lr1 = linear_model.LinearRegression()
lr1.fit(model_1_train, train['price'])
# print lr1.coef_

lr1_poly = linear_model.LinearRegression()
lr1_poly.fit(poly_model_1_train, train['price'])
# print lr1_poly.coef_

lr2 = linear_model.LinearRegression()
lr2.fit(model_2_train, train['price'])
# print lr2.coef_

lr2_poly = linear_model.LinearRegression()
lr2_poly.fit(poly_model_2_train, train['price'])
# print lr2_poly.coef_

lr3 = linear_model.LinearRegression()
lr3.fit(model_3_train, train['price'])
# print lr3.coef_

lr3_poly = linear_model.LinearRegression()
lr3_poly.fit(poly_model_3_train, train['price'])
# print lr3_poly.coef_



pred1 = lr1.predict(model_1_test)
mse1 = mean_squared_error(test['price'], pred1)
pred1_poly = lr1_poly.predict(poly_model_1_test)
mse1_poly = mean_squared_error(test['price'], pred1_poly)
print "Printing MSE of both model_1: "
print mse1, mse1_poly

pred2 = lr2.predict(model_2_test)
mse2 = mean_squared_error(test['price'], pred2)
pred2_poly = lr2_poly.predict(poly_model_2_test)
mse2_poly = mean_squared_error(test['price'], pred2_poly)
print "Printing MSE of both model_2: "
print mse2, mse2_poly

pred3 = lr3.predict(model_3_test)
mse3 = mean_squared_error(test['price'], pred3)
pred3_poly = lr3_poly.predict(poly_model_3_test)
mse3_poly = mean_squared_error(test['price'], pred3_poly)
print "Printing MSE of both model_3: "
print mse3, mse3_poly



score1 = lr1.score(model_1_test, test['price'])
score1_poly = lr1_poly.score(poly_model_1_test, test['price'])
print "Printing SCORE of both model_1:"
print score1, score1_poly

score2 = lr2.score(model_2_test, test['price'])
score2_poly = lr2_poly.score(poly_model_2_test, test['price'])
print "Printing SCORE of both model_2:"
print score2, score2_poly

score3 = lr3.score(model_3_test, test['price'])
score3_poly = lr3_poly.score(poly_model_3_test, test['price'])
print "Printing SCORE of both model_3:"
print score3, score3_poly
