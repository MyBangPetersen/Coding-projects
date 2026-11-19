#this script is a walkthrough of https://github.com/AliHabibnia/Algorithmic_Trading_with_Python/blob/main/Lecture%2005_Time%20Series%20Forecasting.ipynb
#made for my own sake in order to learn more about trading algorithms

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import yfinance as yf
import warnings
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import statsmodels.stats as sms
import scipy.stats as scs
from statsmodels.tsa.stattools import coint, adfuller
warnings.filterwarnings('ignore')

from pandas import Grouper
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import AutoReg

import matplotlib.ticker as ticker

#matplotlib inline
#plt.rc('figure', figsize=(18, 3))
#pd.set_option('display.float_format', lambda x: '%.2f' % x)
#pd.options.display.max_rows = 20

file_io = 'file:///Users/mybangpetersen/Desktop/stockdata.csv'
df = pd.read_csv(file_io)
print(df.head())

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['MSFT'], color='black', label = 'MSFT')
ax.plot(df['Date'], df['AAPL'], color='blue', label = 'AAPL')
ax.plot(df['Date'], df['SBUX'], color='red', label = 'SBUX')
#ax.plot(df['Date'], df['GSPC'], color='green', label = 'GSPC')
ax.plot(df['Date'], df['IBM'], color='orange', label = 'IBM')
space = 700
ax.xaxis.set_major_locator(ticker.MultipleLocator(space))
plt.legend()
plt.show()

#we can have a look at the data through different tools
print('INDEX \n', df.index)
print('SIZE \n', df.size)
print('STATS \n', df.describe())


#looking at just one of the data columns
df = pd.read_csv(file_io, usecols = ['MSFT'])

#the values can be shifted as lag features to predict future values
values = pd.DataFrame(df.values)
df2 = pd.concat([values.shift(2), values.shift(1), values], axis=1)
df2.columns = ['t-2','t-1', 't']
print('LAG FEATURES \n',df2.head())

##the rolling mean is the mean of the previous values
values = pd.DataFrame(df.values)
shifted = values.shift(1)
window = shifted.rolling(window=2)
means = window.mean()

df3 = pd.concat([means, values], axis=1)
df3.columns = ['mean(t-2,t-1)', 't']
print('ROLLING WINDOW \n', df3.head())

#you can also look at the entire data set to keep an eye on bounds
values = pd.DataFrame(df.values)
window = values.expanding()

df4 = pd.concat([window.mean(), values], axis=1)
df4.columns = ['mean', 't']
print('EXPANDING WINDOW \n', df4.head())

#now on for plotting
#we can have a look at the residuals
Re = np.log(df).diff().dropna()
Re.plot()
plt.show()

#or a histogram
df.hist()
plt.show()

#a kernel density (KDE) plot is a smoothed version of a histogram 
df.plot(kind='kde')
plt.show()

##lag plot can show you the correlation relationship
#the correlation is positive or negative depending on the tilt of the data
#this one is positive
lag_plot(df)
plt.show()
#is the data sits tightly at the line the relationship is strong, but if it sits 
#very spread or in a ball in the center there is a weak or no relationship


##we can look at autocorrelation plots (ACF)
#this plots looks at the correlation betwwen the data and its own lag
autocorrelation_plot(df)
plt.show()
#from this we can tell seasonability, trend, mean reversion, randomness, stationairity or model identification

##to remove noise we can use Moving Average Smoothing
rolling = df.rolling(window=3)
rolling_mean = rolling.mean()
rolling_mean.head(10)

#plot original and transformed dataset
plt.plot(df[-100:], label = 'data')
plt.plot(rolling_mean[-100:], label = 'mean')
plt.legend()
plt.show()

##moving average as prediction
#The moving average is a model tha can eaisly be used while taking updates

#prepare problem
X = df.values
window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = []

#walk forward over time steps in test
for t in range(len(test)):
    length = len(history)
    yhat = np.mean([history[i] for i in range(length-window,length)])
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))
rmse = np.sqrt(mean_squared_error(test, predictions))

plt.plot(test)
plt.plot(predictions, color='red')
plt.title('Moving Average')
plt.show()

plt.plot(test[:100])
plt.plot(predictions[:100], color='red')
plt.title('Moving Average (zoomed)')
plt.show()

#all the tools above are usefull when you want to look at the trends and inner workings of a dataset evolving over time
#they can of course be combined to fit the case you are working with best possible

##you can use the Augmented Dickey-Fuller test to see the the data has som sort of trend over time
# calculate stationarity test of return data
X = Re
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
#here we have a statistical value of -11.627 which is less than the values -2,863 at a 5% interval. This means that we will reject the null
#hypothesis with an error lower than 5%, which meansthat the result is statistically unlikely to be a mistake. rejecting the null hypothesis 
#means that the process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.


