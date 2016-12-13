import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import pylab


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

train = pd.read_csv("kc_house_train_data.csv", dtype=dtype_dict)
train = train.sort_values(by=['sqft_living','price'])



### Reshaping data to be (17384, 1) of dimension
shaped_sqft_living = train['sqft_living'].reshape(len(train.index),1)
shaped_price = train['price'].reshape(len(train.index),1)
# print shaped_sqft_living.shape



### Polynomial feature transformation
poly = PolynomialFeatures(degree=15)
poly_sqft_living = poly.fit_transform(shaped_sqft_living)
print poly_sqft_living.shape
# poly_price = poly.fit_transform(shaped_price) # Price (y) should not be polynomial


### Fitting and predicting
lr = linear_model.LinearRegression()
lr.fit(poly_sqft_living, train["price"])
pred = lr.predict(poly_sqft_living)
print pred.shape

plt.plot(train['sqft_living'], train['price'], "x")
# plt.plot(poly_sqft_living, train["price"], ".")
plt.plot(train["sqft_living"], pred, ".")
# plt.axis('equal')
plt.show()

