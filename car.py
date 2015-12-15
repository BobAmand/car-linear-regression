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

plt.scatter(df.Mileage, df.Price)
plt.ylabel("Price")
plt.xlabel("Mileage")
plt.title("Price x Mileage for Used Cars")
plt.plot(df[['Mileage']],
         regr1.predict(df[['Mileage']]),
         color='red',
         linewidth=2)
plt.show()

print("Mileage alone is not a good predictor of price.")
print("The Rsq at {0:.2f} is very low. Only {1:.2f} of Price variance is captured in Mileage.".
      format(regr1.score(df[['Mileage']], df[['Price']]),
             regr1.score(df[['Mileage']], df[['Price']])))
print("The current model is insufficient to predict price.")
