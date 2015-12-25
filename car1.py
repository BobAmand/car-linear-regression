import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import itertools


fulldf = pd.read_csv("car_data.csv")
possible_columns = ['Mileage', 'Cylinder', 'Liter',
                    'Doors', 'Cruise', 'Sound', 'Leather']


def high_value_regression(fulldf, possible_columns):

    '''
combos=[]
- the range "1" allows control of the minimum number of columns to be included.
- The 'intertools' creates a list of tuples for each field combination.
- Each tuple is unpacked to a list prior to evaluation by LinearRegression().
    '''
    combos = []
    for x in range(1, len(possible_columns)):
        combos.append(list(itertools.combinations(possible_columns, x)))
    print(len(possible_columns))
    # print(combos)
    # print %timeit
    '''
letter=[]
   - Each tuple from 'combo' is unpacked to a list and appended into letter.
    '''
    letter = []
    low = 1
    high = 126
    for x in combos:
        for y in x:
            letter.append(list(y))
    print("printing letter list:")
    print(len(combos))
    for s in range(low, high):
        print(letter[s]),

    '''
results=[]
   - The list of column names are plugged into df[x] for regression.
   - After each loop through the list of lists:
       o The 'results' list accummulates two columns: 'Grouping' and 'Score'.
       o The list of column names are 'join'd and appended into 'Grouping'.
       o The Rsq is recorded as 'Score' in the DataFrame.
   - The table is sorted high to low and 5 top printed.

    '''
    # results = []
    # output = fulldf['Price']
    # for x in letter:
    #     input_data = fulldf[x]
    #
    #     regrm = linear_model.LinearRegression()
    #     regrm.fit(input_data, output)
    #     regrm.coef_
    #     regrm.intercept_
    #
    #     results.append([', '.join(x), regrm.score(input_data, output)])
    #
    # results = pd.DataFrame(results, columns=('Grouping', 'Score'))
    #
    # return results.sort_index(by='Score', ascending=False).head(5)
    # print(results)
    # print(output)
    # print(fulldf)

high_value_regression(fulldf, possible_columns)

# print("Oddly, cannot discover why the initial 7 column Rsq (including 'liter) = higher than max in table!!")
# print("Rsq all 7 = 0.44626")
