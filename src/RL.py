                              Source Code No - 5 

########### Source code for Reinforcement Learning #######################

'''---------Libraries---------'''

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pyexcel
import math as m

'''---------Libraries---------'''

############ Variable Initialization####################################
df             = pd.read_csv("data/Finance/LIC.csv")
df             = df.fillna(method='bfill')
test_len       = 1000
portfolio      = [10000,0,0]
portfolio_test = [10000,0,0]
epochs         = 2000
gamma          = 0.985 
epsilon        = 1
index          = 15
reward_extent  = 99
no_of_states   = 3*3*3*3
data           = [['cash', 'no_of_stocks','stock_val',
                   'action','betaq','reward',
                   'todaysprice','portfolio','state']]
data_test      = [['cash', 'no_of_stocks','stock_val',
                   'action','reward','todaysprice',
                   'portfolio','state']] 
maxc = 100
minc = -100
#####################################################################

############# Initializing Q table #################################
Q = np.random.uniform(low=0, high=10 , size=(no_of_states,3))
####################################################################        

#######################Function Definitions##########################

'''--------Funtion to return a dataframe for given range of dates---------'''
def get_data(start_date, end_date, ticker):
    dates   = pd.date_range(start_date, end_date)  
    df      = pd.DataFrame(index=dates)
    df_temp = pd.read_csv('data/' + ticker + '.csv',
             index_col='Date', parse_dates=True, na_values=['nan'])
    df      = df.join(df_temp)
    df      = df.fillna(method='ffill')
    return df

'''-------Function to get value of stock price at given index--------'''
def get_value(dataFrame, index):
    return dataFrame.get_value(index, 'Adj Close')

'''-------Function to get value of rolling mean for a given window size of dataFrame------'''
def get_rolling_mean(values, window):
    return values.rolling(window=window, center=False).mean()

'''-------Function to get rolling standard deviation for a given window size of dataFrame------'''
def get_rolling_std(values, window):
    return values.rolling(window=window, center=False).std()

'''-------Function to get exponential moving average for a given window size of dataFrame------'''
def get_exp_moving_avg(dataFrame, window):
    return dataFrame.ewm(span=window).mean()

'''-------Function to get momentum of stockprice for a given dataFrame--------'''
def get_momentum(dataFrame, length):
    momentum = dataFrame.diff(periods=length)
    return momentum

'''-------Function which returns trend direction--------'''
def get_trend_direction(df):
    dfm       = df.rolling(window=moving_avg_width, center=False).mean()
    slope     = get_momentum(dfm, 1)
    slope_arr = np.array(slope)
    trend_dir = np.average(slope_arr[moving_avg_width:(moving_avg_width+data_width-1)])
    return trend_dir


'''--------Function which gives ADl for a given dataFrame-------'''
def get_adl(dataFrame):
    money_flow_multiplier  = []
    money_flow_volume      = []
    prev_Adl               = 0
    Adl_list               = []

    for i in range(0, len(df)):
        if (dataFrame.get_value(i, 'High')-dataFrame.get_value(i, 'Low')) != 0:
            df_Close       = dataFrame.get_value(i, 'Close')
            df_Low         = dataFrame.get_value(i, 'Low')
            df_High        = dataFrame.get_value(i, 'High')
            
            
            money_flow_mul = ((df_Close - df_Low)- (df_High - df_Close))/(df_High-df_Low)
            money_flow_multiplier.append(money_flow_mul)
            money_flow_volume.append(money_flow_mul*dataFrame.get_value(i, 'Volume'))
            Adl            = prev_Adl + money_flow_mul*dataFrame.get_value(i, 'Volume')
        else:
            Adl            = prev_Adl

        prev_Adl = Adl
        Adl_list.append(Adl)
             
    return Adl_list


'''--------Function which gives RSI for a given dataFrame--------'''
def get_RSI(df):
    current_data = df['Adj Close']
    RSI          = [0] * len(df)

    for i in range(15, len(df)):
        data_req = current_data[i-15:i]
        change   = data_req.diff()
        data_len = len(change)
        change   = change.fillna(method='bfill')
        change   = np.array(change)
        posGain  = 0
        negLoss  = 0

        for j in range(data_len):
            if change[j] >= 0:
                posGain   = posGain + change[j]
            else:
                negLoss   = negLoss - change[j]
        
        if  negLoss      == 0:
            RSI[i]        = 100
        else:
            gtol_ratio    = posGain/negLoss
            RSI[i]        = 100 - (100/(1 + gtol_ratio))
    return RSI


'''--------Function which gives Bollinger bands for a given dataFrame--------'''
def get_bollinger_bands(dataFrame):
    rolling_mean            = get_rolling_mean(dataFrame['Adj Close'], 20)
    rolling_std_dev         = get_rolling_std(dataFrame['Adj Close'], 20)

    upper_bollinger_band    = rolling_mean + 2 * rolling_std_dev
    lower_bollinger_band    = rolling_mean - 2 * rolling_std_dev

    return upper_bollinger_band, lower_bollinger_band


'''--------Function which returns MACD for a given dataFrame--------'''
def get_MACD(df):
    ema_12 = get_exp_moving_avg(df['Adj Close'], 12)
    ema_26 = get_exp_moving_avg(df['Adj Close'], 26)
    MACD_line = ema_12 - ema_26
    signal_line = get_exp_moving_avg(MACD_line, 9)
    MACD_hist = MACD_line - signal_line

    return MACD_hist

'''-------Function which Returns CCI for a given dataFrame---------'''
def get_cci(df):
    CCI           = [0] * len(df)
    typical_price = (df['High'] + df['Close'] + df['Low']) / 3

    for i in range(15, len(typical_price)):
        tp_req         = typical_price[i-15:i]
        tp_req         = np.array(tp_req)
        sma            = np.average(tp_req)
        sma            = [sma] * 15
        mean_deviation = np.sum(abs(tp_req - sma))/14
        CCI[i]         = (tp_req[14] - np.average(tp_req)) / (0.015 * mean_deviation)

    return CCI

'''----------- Function which returns State -------------'''
def get_state(index):
    global df
    global maxc
    global minc

    ub, lb     = get_bollinger_bands(df)
    bc         = 0
    mh         = get_MACD(df,index)
    c          = get_cci(df,index)
    cc         = 0
    rc         = 0
    mhc        = 0
    r          = get_RSI(df,index)

    if c >= 100:
           if c  >= 100:
               cc   = 2
               maxc = c
           elif c <= 100:
               cc   = 0
    elif c <= -100:
           if c <= -100:
               cc   = 1
               minc = c
           elif c >= -100:
               cc   = 0
    else:
           maxc     = 100
           minc     = -100

    if r > 70:
           rc       = 2
    elif r < 30:
           rc       = 1

    if (mh < 0) and (get_MACD(df,index-1) > 0):
           mhc = 2
    elif (mh > 0) and (get_MACD(df,index-1) < 0):
           mhc = 1
    if current_data[index] >= ub[index]:
           bc = 2
    elif current_data[index] <= lb[index]:
           bc = 1
    return [cc,bc,mhc,rc]

'''-----------Function which returns action depending upon current state------'''
def get_action(state):
    global Q
    global epsilon
    epsilon   = float(epsilon) - 1.0/epochs
    
    index     = 3*3*3*state[0] + 3*3*state[1] + 3*state[2] + state[3]
    qval      = Q[index,:]
    qvalt     = qval + [min(qval), min(qval), min(qval)]
    betaq     = max(qvalt)/(qvalt[1]+qvalt[2]+qvalt[0])
    
    if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,3)
    else:                           #choose best action from Q(s,a) values
            action = (np.argmax(qval))
    return action,betaq


def updateQ(state , action , reward , next_state):
    global Q
    learning_rate    = 0.5
    discount_factor  = 0.8
    index            = 3*3*3*state[0] + 3*3*state[1] + 3*state[2] + state[3]

    index_next       = 3*3*3*next_state[0] + 3*3*next_state[1] 
                       + 3*next_state[2] + next_state[3]
    Q[index, action] = (1 - learning_rate)*Q[index , action] + learning_rate*(reward + discount_factor*max(Q[index_next , :]))
        
def get_reward(state , action , betaq, portfolio, iter):
  
  portfolio_prev = portfolio
  global df
  global window
  todaysprice  = get_value(df,iter)
  portfolio_ab = [0,0,0]
  portfolio_as = [0,0,0]
  val_ab       = 0 #assumed buy
  val_as       = 0 #assumed sell

###############################################################
#  actions - 0 -> hold , 1 -> buy , 2-> sell                  #
# portfolio -[cash balance , no. of stocks , stock valuation] #
###############################################################
  
  if action==1:
      if(betaq*portfolio[0]>todaysprice):
          stockNo   = portfolio[1]+int(portfolio[0]*betaq/todaysprice)
          stockVal  = stockNo*todaysprice
          cashVal   = portfolio[0] - int(portfolio[0]*betaq/todaysprice)*todaysprice
          portfolio = [cashVal, stockNo, stockVal]
          reward    = ((portfolio[0]+portfolio[1]*get_value(df,iter+reward_extent))
                     -((portfolio_prev[0]+portfolio_prev[1]*get_value(df,iter+reward_extent))))
          if(reward<0):
            reward  = -m.pow(abs(reward), float(1)/3 )
          else:
            reward  = reward**(1/3.0) 
      else:
          reward,portfolio = get_reward(state,0,betaq,portfolio,iter)
      
  if action==2:
       if(portfolio[1]<3):
          reward,portfolio = get_reward(state,0,betaq,portfolio,iter)
       else:
          stockNo     = portfolio[1] - int(portfolio[1]*betaq)
          stockVal    = stockNo*todaysprice
          cashVal     = portfolio[0]+ (int(portfolio[1]*betaq)*todaysprice)
          portfolio   = [cashVal, stockNo, stockVal]
          reward      = ((portfolio[0]+portfolio[1]*get_value(df,iter+reward_extent))
                       -((portfolio_prev[0]+portfolio_prev[1]*get_value(df,iter+reward_extent))))
          if(reward<0):
              reward  = -m.pow(abs(reward), float(1)/3 )
          else:
              reward  = reward**(1/3.0)

  
  if action==0:
      if(betaq*portfolio[0]>todaysprice):
          stockNo_ab   = portfolio[1]+int(portfolio[0]*betaq/todaysprice)
          stockVal_ab  = stockNo_ab*todaysprice
          cashVal_ab   = portfolio[0] - int(portfolio[0]*betaq/todaysprice)*todaysprice
          portfolio_ab = [cashVal_ab, stockNo_ab, stockVal_ab] 
      else:
          portfolio_ab = [0,0,0]
       
      if(portfolio[1]<3):
         portfolio_as =  [0,0,0]
      
      else:
        stockNo_as     = portfolio[1] - int(portfolio[1]*betaq)
        stockVal_as    = stockNo_as*todaysprice
        cashVal_as     = portfolio_as[0]+ (int(portfolio[1]*betaq)*todaysprice)
        portfolio_as   = [cashVal_as, stockNo_as, stockVal_as]
        
      val_as           = portfolio_as[0]+portfolio_as[1]*get_value(df,iter+reward_extent)  
      val_ab           = portfolio_ab[0]+portfolio_ab[1]*get_value(df,iter+reward_extent)

      portfolio        = [portfolio[0], portfolio[1], portfolio[1]*todaysprice]
      
      reward           = ((portfolio[0]+portfolio[1]*get_value(df,iter+reward_extent))
                        -((max(val_as,val_ab))))-2
      if(reward<0):
              reward   = -m.pow(abs(reward), float(1)/3 )
      else:
              reward   = reward**(1/3.0)     

  return reward,portfolio    
  
#################training##################################################

for j in range(25,epochs-1):
        
    state            = get_state(j)
    next_state       = get_state(j+1)
    
    action,betaq     = get_action(state)
    reward,portfolio = get_reward(state,action,betaq,portfolio,j)
    
    reward0,port0    = get_reward(state,0,betaq,portfolio,j)
    reward1,port1    = get_reward(state,1,betaq,portfolio,j)
    reward2,port1    = get_reward(state,2,betaq,portfolio,j)
    
    updateQ(state,0,reward0,next_state)
    updateQ(state,1,reward1,next_state)
    updateQ(state,2,reward2,next_state)

    data.append([portfolio[0],
                portfolio[1],
                portfolio[2],
                action,
                betaq,
                reward,
                get_value(df,j),
                portfolio[0]+portfolio[2],
                state])

print ("writing train")
pyexcel.save_as(array = data, dest_file_name = 'data_train.csv')
print ("training complete")
######################################################################
print ("Now testing")

net_worth         = [0] * test_len
total_stocks      = [0] * test_len
portfolio_test[2] = get_value(df,epochs+1)*portfolio_test[1]

for j in range(epochs,epochs+test_len):
    
    net_worth[j-epochs-1]      = portfolio_test[0] + portfolio_test[2]
    total_stocks[j-epochs-1]   = portfolio_test[1]
    state_test                 = get_state(j)
    action_test, betaq_test    = get_action(state_test)
    reward_test,portfolio_test = get_reward(state_test, action_test,
                                            betaq_test, portfolio_test,j)

    data_test.append([portfolio_test[0],
                      portfolio_test[1],
                      portfolio_test[2],
                      action_test,
                      betaq_test,
                      reward_test,
                      get_value(df,j),
                      portfolio_test[0]+portfolio_test[2],
                      state_test])
    pyexcel.save_as(array = data_test, dest_file_name = 'data_test.csv')

############ Plotting Portfolio #############################################
plt.subplot(211)
df['Adj Close'].plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.xlabel('Time')
plt.ylabel('Adjusted Close of Price')
plt.subplot(212)
plt.plot(net_worth)
plt.xlabel('Time')
plt.ylabel('Net Worth')
plt.show()

##############################################################################

   
    



