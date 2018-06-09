import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math as m


def mean_of_dailyreturns(daily_returns):
    return daily_returns.mean()


def variance_of_dailyreturns(daily_returns):
    return np.var(daily_returns)


def std_deviation_of_dailyreturns(daily_returns):
    return np.std(daily_returns)


def get_dailyreturns(df):
    dr = [0.0] * len(df)
    dr = np.array(dr)
    df = np.array(df)

    for i in range(1, len(df)):
        dr[i - 1] = ((df[i] - df[i - 1]) / df[i - 1])
    return dr


def get_data(start_date, end_date, ticker):
    dates = pd.date_range(start_date, end_date)  # '2010-12-31'
    df = pd.DataFrame(index=dates)
    df_temp = pd.read_csv('data/' + ticker + '.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
    df = df.join(df_temp)
    df = df.dropna()
    return df


def mean_variance(cov, ret, wts):
    weights = np.array([wts])
    mn = np.matmul(ret, np.transpose(weights))
    s = 0
   
    for l in range(0, stock_no):
        for ma in range(0, stock_no):
            s += cov[l][ma] * weights[0][l] * weights[0][ma]
    '''
    temp = np.matmul(weights , cov)
    s= np.matmul(temp , np.transpose(weights))
    '''
   
    return s, mn


tickers = ['Banks/ICICI', 'IT/TCS', 'Pharma/Sun_Pharma', 'Finance/LIC', 'Refineries/BPCL', 'Infra/ABB', 'Telecom/Airtel', 'IT/Infosys']
all_mean = np.array([0.0] * len(tickers))
all_std = np.array([0.0] * len(tickers))
all_var = np.array([0.0] * len(tickers))
s_date = '2017-01-02'
e_date = '2017-03-24'
dates = pd.date_range(s_date, e_date)
all_daily_returns = []
stock_no = len(tickers)
wts = np.array([(1.0 / stock_no)] * stock_no)
final_weights = [(1.0 / stock_no)] * stock_no
Budget = 1
rho = 0.2

for i in range(len(tickers)):
    df_test = get_data(s_date, e_date, tickers[i])
    daily_ret = get_dailyreturns(df_test)
    all_daily_returns.append(daily_ret)
    all_mean[i] = mean_of_dailyreturns(daily_ret)
    all_std[i] = std_deviation_of_dailyreturns(daily_ret)
    all_var[i] = variance_of_dailyreturns(daily_ret)

all_daily_returns = np.transpose(all_daily_returns)


def func_cost(a, b, c, d, e, f, g, h):
    global Rh
    global final_weights
    all_weights = []
    weights = [a, b, c, d, e, f, g, h]
    for j in range(len(all_daily_returns)):
        all_weights.append(weights)
    
    Rhcols = np.multiply(all_daily_returns, all_weights)
    Rh = np.sum(Rhcols, 1)
    B = [rho * Budget]*len(Rh)
    cost = sum((Rh - B)**2)
    #print cost
    #print weights
    final_weights = weights
    return cost

print tickers
print 'Compound return', all_mean*100
print 'Risk           ', all_std

Rh = [1] * len(all_daily_returns)
iter_no = 8000

bnds = ((0, Budget), (0, Budget), (0, Budget), (0, Budget), (0, Budget), (0, Budget), (0, Budget), (0, Budget))

cons = ({'type': 'eq', 'fun': lambda x:  Budget - (x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7])},
        {'type': 'ineq', 'fun': lambda x:  rho*Budget - np.mean(Rh)},
        )

x = minimize(lambda x: func_cost(x[0], x[1], x[2], x[3] , x[4], x[5], x[6], x[7]), wts, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter':iter_no})

#sigma = np.cov(all_daily_returns)

sigma = [[0]*stock_no for i in range(stock_no)]
drt = np.transpose(all_daily_returns)

for b in range(0, stock_no):
    for h in range(0, stock_no):
        sigma[b][h] = sum(np.multiply(all_daily_returns[b], all_daily_returns[h]))
        #print sum(np.multiply(all_daily_returns[b] , all_daily_returns[h]))

mu = np.array([all_mean])
#sigma = np.cov(np.transpose(all_daily_returns))
sigma_port, mu_port = mean_variance(sigma, mu, final_weights)

print 'Final weights    ', final_weights*100
print 'Final return     ', mu_port*100
print 'Portfolio Risk   ', m.sqrt(sigma_port)

for i in range(0, stock_no):
    plt.plot(all_std[i], all_mean[i]*100, 'ro')
    plt.text(all_std[i], all_mean[i]*100, tickers[i], horizontalalignment='center')

plt.plot(m.sqrt(sigma_port), mu_port*100, 'go', label='Risk vs Return \n rho = ' + str(rho))
plt.xlabel('Risk')
plt.ylabel('Return')
plt.text(m.sqrt(sigma_port), mu_port*100, 'Portfolio', horizontalalignment='center')
plt.legend(bbox_to_anchor=(1.05, 1), loc='lower right', borderaxespad=0.)

plt.show()
