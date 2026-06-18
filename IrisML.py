import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset from my desktop and skipping empty spaces
file_path = '/Users/mybangpetersen/Desktop/iris.csv'
df = pd.read_csv(file_path, skipinitialspace=True)
df.columns = df.columns.str.strip()
#print below to have a look at the numbers
#print(df)

#plotting some of the data
fig, ax = plt.subplots()
ax.scatter(
    df['SepalLengthCm'],
    df['SepalWidthCm'],
    #each flower should have its own color
    c=df['Species'].map({
        'Iris-setosa': 'red',
        'Iris-versicolor': 'blue',
        'Iris-virginica': 'green',
    }),
)
ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('SepalWidthCm')
plt.show()
