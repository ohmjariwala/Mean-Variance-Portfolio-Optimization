# ## Mean Variance Portfolio Optimization

# Where $\Sigma^{-1}$ is the inverse of the covariance matrix, and the matrix of 1s has n elements, where n is the number of assets.

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta

myPortfolio = ["AAPL", "JPM", "GE", "DIS"]

def MinVarWeights(tickers: list):
    startDate = str(date.today() - timedelta(days = 365))
    endDate = str(date.today()) #Take today's date
    data = yf.download(myPortfolio, start = startDate , end = endDate , progress = False)['Adj Close'].pct_change().dropna()
    cov = data.cov() #Find the covariance matrix
    inv = np.linalg.inv(cov) #Take the inverse of the covariance matrix
    ones = np.ones(len(myPortfolio)) #Create identity matrix of length n
    
    minvar = (np.dot(inv, ones))/(np.dot(np.dot(ones.T, inv), ones)) #Find the min variance
    return minvar

print(MinVarWeights(myPortfolio))
