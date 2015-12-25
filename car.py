import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import itertools

# % matplotlib inline

df = pd.read_csv("car_data.csv")
df.head(3)

regr1 = linear_model.LinearRegression()
regr1.fit(df[['Mileage']], df[['Price']])
print("The x coefficient = {}".format(regr1.coef_))
print("  The y intercept = {}".format(regr1.intercept_))

print("\n          The Rsq = {}".format(regr1.score(df[['Mileage']],
                                                    df[['Price']])))

'''
The plot interupts the flow so 'commented-out' for now.
'''

# plt.scatter(df.Mileage, df.Price)
# plt.ylabel("Price")
# plt.xlabel("Mileage")
# plt.title("Price x Mileage for Used Cars")
# plt.plot(df[['Mileage']],
#          regr1.predict(df[['Mileage']]),
#          color='red',
#          linewidth=2)
# plt.show()

print("Mileage alone is not a good predictor of price.")
print("Only {0:.6f} of Price variance is captured in Mileage.".
      format(regr1.score(df[['Mileage']], df[['Price']])))
print("The current model is insufficient to predict price.\n")

fulldf = df[['Price', 'Mileage', 'Cylinder', 'Liter',
             'Doors', 'Cruise', 'Sound', 'Leather']]
input = fulldf[['Mileage', 'Cylinder', 'Liter',
                'Doors', 'Cruise', 'Sound', 'Leather']]
output = fulldf['Price']

regrm = linear_model.LinearRegression()
regrm.fit(input, output)

print("Coefficients for ")
print("Mileage, Cylinder, Liter, Doors, Cruise, Sound, leather:")

print(regrm.coef_)
print("\n")
print("Y-intercept = {0:.4f}".format(regrm.intercept_))
print("    The Rsq = {0:.4f}".format(regrm.score(input, output)))

''' Full correlation '''

fulldf.corr()

'''Based on highest correlation with Price '''

subdf = df[['Price', 'Mileage', 'Cylinder', 'Cruise']]

input = fulldf[['Mileage', 'Cylinder', 'Cruise']]
output = fulldf['Price']

regrm = linear_model.LinearRegression()
regrm.fit(input, output)
print("\n")
print("Coefficients for Mileage, Cylinder + Cruise:")
print("\n")
print(regrm.coef_)
print("Y-intercept = {0:.4f}".format(regrm.intercept_))
print("    The Rsq = {0:.4f}".format(regrm.score(input, output)))
