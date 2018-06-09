
                              Source Code No - 4

########### Source code for Branching Model of Algorithmic Trading ###########  

'''---------Libraries---------'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyexcel as pyex
import math as m
'''---------Libraries---------'''


df = pd.read_csv('data/Auto/Ashok_leyland.csv')  #Reading a Dataframe



#######################Function Definitions###################################

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


'''------Function which Returns TrueRange of the given dataFrame------''' 
def truerange(matrix):
    zeros                = [0] * len(df)
    true_range           = [0] * len(matrix)
    HL                   = matrix[:, 0] - matrix[:, 1]
    new_matrix           = np.hstack([np.transpose([HL.ravel()]),
                                      np.transpose([zeros]), np.transpose([zeros])])
    for i in range(1, len(matrix)):
        new_matrix[i, 1] = abs(matrix[i, 0] - matrix[(i - 1), 2])
        new_matrix[i, 2] = abs(matrix[i, 1] - matrix[(i - 1), 2])

    for i in range(0, len(true_range)):
        true_range[i]    = np.max(new_matrix[i, :])

    return true_range


'''---------Function which returns Directional indicators of the given dataFrame-------- '''
'''--------------------------ADX, DX, PDI, MDI------------------------------------------''' 
def get_directional_indicators(df):
    pdm       = [0]*len(df)
    mdm       = [0]*len(df)
    high      = df['High']
    low       = df['Low']
    close     = np.array([df['Close']])
    look_back = 13
    mat = np.hstack([np.transpose([np.array(high)]),
                     np.transpose([np.array(low)]), 
                     np.transpose(close)])

    for i in range(1, len(df)):
        upmove     = high[i] - high[i-1]
        downmove   = low[i-1] - low[i]
        if (upmove > downmove) and (upmove >0):
            pdm[i] = upmove
        if (downmove >upmove) and (downmove >0):
            mdm[i] = downmove

    tr                     = truerange(mat)
    tri14                  = [0]*len(df)
    pdm14                  = [0]*len(df)
    pdm14[look_back + 1]   = pdm14[look_back + 1]/14
    pdm14[look_back + 2]   = (((look_back + 1)*pdm14[look_back + 1])/
                              (look_back + 1) + pdm[look_back + 2])
    mdm14                  = [0]*len(df)
    mdm14[look_back + 1]   = mdm14[look_back + 1]/14
    mdm14[look_back + 1]   = np.sum(mdm[look_back + 1:0])/14
    mdm14[look_back+2]     = ((look_back*mdm14[look_back+1])/
                             (look_back+1) + mdm[look_back+2])
    pdi14                  = [0]*len(df)
    mdi14                  = [0]*len(df)
    DX                     = [0] * len(df)
    DX                     = np.abs(np.array(DX))
    data_req               = DX[(look_back+1):(look_back+1)*2]
    ADX                    = [0] * len(DX)
    ADX[(look_back + 1)*2] = np.mean(data_req)
    tri14[look_back + 1]   = tri14[look_back + 1]/14
    tri14[look_back + 2]   = ((look_back*tri14[look_back + 1])/
                              (look_back + 1) + tr[look_back + 2])

    for i in range(0, look_back+1):
        tri14[look_back + 1] += tr[i]

    for i in range(look_back + 3, len(df)):
        tri14[i]              = (look_back*tri14[i-1]/(look_back + 1) + tr[i])

    for i in range(0, look_back+1):
        pdm14[look_back + 1] += pdm[i]

   
    for i in range(look_back + 3, len(df)):
        pdm14[i]              = ((look_back*pdm14[i-1])/(look_back + 1) + pdm[i])

    for i in range(0, look_back+1):
        mdm14[look_back + 1] += mdm[i]

    for i in range(look_back+3,  len(df)):
        mdm14[i]              = ((look_back*mdm14[i-1])/(look_back + 1) + mdm[i])

    for i in range(look_back+1, len(df)):
        pdi14[i]              = 100*np.divide(pdm14[i], tri14[i])
        mdi14[i]              = 100*np.divide(mdm14[i], tri14[i])
        DX[i]                 = 100*(pdi14[i] - mdi14[i])/(pdi14[i] + mdi14[i])

    for i in range((look_back+1)*2 + 1, len(DX)):
        ADX[i]                = (ADX[i - 1]*look_back + DX[i])/(look_back+1)

    return ADX, DX, pdi14, mdi14


####################### End Of Function Definitions###################################

####################### Defining Variables ##########################################


'''------Initializing coloumns for Excel file----------'''

data          = [['Cash_val',  'no_of_stocks',  'todaysprice',  'stock_val',
                  'portfolio', 'Shares bought', 'Amt bought',   'Shares sold',
                  'Amt Sold',  'Buy',           'Sell',         'Hold']]
trend_data    = [['RSI', 'MACD', 'BB',
                  'ADL', 'Percent_profit', 'daily_ret portfolio',
                  'Reward_buy', 'Reward_for_sell', 'Reward_for_hold']]
no_trend_data = [['Support', 'Resistance', 'Percent Profit', 
                  'daily_ret portfolio', 'Reward_buy', 'Reward_sell', 'Reward_hold']]

data_width      = 10   
moving_avg_width= 5 
portfolio       = [10000, 0, 0] #Portfolio = [Cash_value,No_Of_Stocks,Todays_Stock_Price]

buysell         = [0]*len(df)
buysell1        = [0]*len(df)
buy             = [0]*len(df)
sell            = [0]*len(df)
hold            = [0]*len(df)
Amt_b           = [0]*len(df)
Amt_s           = [0]*len(df)
shares_b        = [0]*len(df)
shares_s        = [0]*len(df)

rsi             = [0]*len(df)
macd            = [0]*len(df)
bb              = [0]*len(df)
ad_line         = [0]*len(df)
percent_profit  = [0]*len(df)
sup             = [0]*len(df)
res             = [0]*len(df)
daily_ret_port  = [0]*len(df)

rew_b           = [0]*len(df)
rew_s           = [0]*len(df)
rew_h           = [0]*len(df)

current_data    = df['Adj Close']
################### End of variable definitions############################


###################Calculation of Technical Factors######################
ADX, DX, pdi, mdi       = get_directional_indicators(df)
directional_index       = [0]*len(df)

upper_band, lower_band  = get_bollinger_bands(df)
bollinger_index         = [0]*len(df)

MACD                    = get_MACD(df)
MACD_var                = [0]*len(df)

CCI                     = get_cci(df)
CCI_var                 = [0]*len(df)
maxc                    = 100
minc                    = -100

RSI                     = get_RSI(df)
RSI_var                 = [0]*len(df)

adl                     = pd.DataFrame(get_adl(df))
adl_max                 = np.max(np.array(abs(adl)))
sp                      = df['Adj Close']
pow_n                   = int(m.log(adl_max))
stock_price_max         = np.max(np.array(sp))
pow_d                   = int(m.log(stock_price_max))
gamma                   = pow(10,pow_n)/pow(10,(pow_n-pow_d))
adlc                    = [0]*len(df)

n                       = 500
portfolio[2]            = portfolio[1]*get_value(df, n-1)
net_worth               = [portfolio[0]+portfolio[2]]*len(df)
exit_flag               = 1
effective_buying_price  = get_value(df, n-1)
trend_angle_adl         = 0
trend_angle_data        = 0
factor                  = 0

#################End of Calculation of Technical Factors#################

#################Iterating over the DataFrame for stock trading###############
for i in range(n, len(df)):
    todaysprice         = get_value(df, i)
    net_worth[i]        = (portfolio[0] + portfolio[1] * todaysprice)
    daily_ret_port[i]   = ((net_worth[i] - net_worth[i - 1]) / net_worth[i - 1]) * 100
    percent_profit[i]   = ((todaysprice - effective_buying_price)/effective_buying_price)*100

########For Trending Market################
    if (ADX[i] > 20) and exit_flag: 
        
        if (RSI[i-1] > RSI[i])   and (RSI[i] >= 70):
            RSI_var[i]  = -10
        elif (RSI[i-1] < RSI[i]) and (RSI[i] <= 30):
            RSI_var[i]  = 10
        rsi[i] = RSI[i]

        if (MACD[i] < 0)         and (MACD[i-1] > 0):
            MACD_var[i] = -5
            macd[i]     = MACD[i] - MACD[i-1]
        elif (MACD[i] > 0)       and (MACD[i-1] < 0):
            MACD_var[i] = 5
            macd[i]     = MACD[i] - MACD[i-1]

        if current_data[i]  >= upper_band[i]:
            bollinger_index[i]    = -10
            bb[i]                 = current_data[i] - upper_band[i]
        elif current_data[i]<= lower_band[i]:
            bollinger_index[i]    = 10
            bb[i]                 = current_data[i] - lower_band[i]
            temp1                 = adl[(i-moving_avg_width-data_width):i]/gamma
            temp2                 = current_data[(i-moving_avg_width-data_width):i]
            trend_angle_adl       = get_trend_direction(temp1)
            trend_angle_data      = get_trend_direction(temp2)

        if (trend_angle_adl > 1) and (trend_angle_data < 0):
            adlc[i]               = 5
            ad_line[i]            = trend_angle_adl
        elif (trend_angle_adl < -1) and (trend_angle_data > 0):
            adlc[i]               = -5
            ad_line[i]            = trend_angle_adl


        if (pdi[i-1] > mdi[i-1]) and (pdi[i] < mdi[i]):
            directional_index[i]  = -5
        elif (pdi[i-1] < mdi[i-1]) and (pdi[i] > mdi[i]):
            directional_index[i]  = 5

        ####### Updating buysell score for each Factor########
        buysell = np.add(CCI_var, RSI_var)
        buysell = np.add(buysell, bollinger_index)
        buysell = np.add(buysell, MACD_var)
        buysell = np.add(buysell, adlc)
        buysell = np.add(buysell, directional_index)

        ####### Taking suitable action from buysell ##########
        
        ####### Action of buying stocks ###############
        if buysell[i] > 0:
            if buysell[i]   == 5:
                factor = 0.5
                buy[i] = 5
                print i, "Buy very Less"

            elif buysell[i] == 10:
                factor = 0.6
                buy[i] = 10
                print i, "Buy Less"

            elif buysell[i] == 15:
                factor = 0.7
                buy[i] = 15
                print i, "Buy Moderate"

            elif buysell[i] == 20:
                factor = 0.8
                buy[i] = 20
                print i, "Buy More"

            elif buysell[i] > 20:
                factor = 0.9
                buy[i] = 30
                print i, "Buy extreme"

            if portfolio[0] < todaysprice:
                shares_b[i]            = 0
                Amt_b[i]               = 0
                effective_buying_price = (effective_buying_price * portfolio[1] + Amt_b[i])/(portfolio[1] + shares_b[i])
                stockNo                = portfolio[1]
                stockVal               = stockNo * todaysprice
                cashVal                = portfolio[0]
            else:
                shares_b[i]            = int(portfolio[0] * factor / todaysprice)
                Amt_b[i]               = shares_b[i] * todaysprice
                effective_buying_price = (effective_buying_price*portfolio[1] + Amt_b[i])/(portfolio[1] + shares_b[i])
                stockNo                = portfolio[1] + shares_b[i]
                stockVal               = stockNo * todaysprice
                cashVal                = portfolio[0] - Amt_b[i]

            portfolio = [cashVal, stockNo, stockVal]
            rew_b[i]  = buy[i]*10
            rew_s[i]  = -buy[i]*10

        ####### Action of Selling stocks ###############
        elif buysell[i] < 0 and percent_profit[i] > 0 and daily_ret_port[i] > 0:
            if buysell[i]   == -5:
                factor  = 0.1
                sell[i] = 5
                print i, "Sell very Less"

            elif buysell[i] == -10:
                factor  = 0.3
                sell[i] = 10
                print i, "Sell Less"

            elif buysell[i] == -15:
                factor  = 0.8
                sell[i] = 15
                print i, "Sell moderate"

            elif buysell[i] == -20:
                factor  = 0.9
                sell[i] = 20
                print i, "Sell More"

            elif buysell[i] < -20:
                factor  = 0.9
                sell[i] = 30
                print i, "Sell extreme"

            if portfolio[1] == 1:
                shares_s[i] = 1
                Amt_s[i]    = todaysprice
                stockNo     = 0
                stockVal    = stockNo * todaysprice
                cashVal     = portfolio[0] + Amt_s[i]
            else:
                shares_s[i] = int(portfolio[1] * factor)
                Amt_s[i]    = shares_s[i] * todaysprice
                stockNo     = portfolio[1] - shares_s[i]
                stockVal    = stockNo * todaysprice
                cashVal     = portfolio[0] + Amt_s[i]

            portfolio  = [cashVal, stockNo, stockVal]
            rew_s[i]   = sell[i]*10 + 20*percent_profit[i]
            rew_b[i]   = -sell[i]*10 - 20*percent_profit[i]

        ####### Action of Holding stocks ###############
        else:
            portfolio = [portfolio[0], portfolio[1], portfolio[1] * todaysprice]
            hold[i]   = 10
            rew_h[i]  = hold[i]*10
            print i, "Hold in trend"

        ######## Appending calculated factors in a csv file ############
        trend_data.append([rsi[i], macd[i], bb[i], ad_line[i], percent_profit[i], daily_ret_port[i], rew_b[i], rew_s[i], rew_h[i]])

    ######### For Sideways Market #########
    elif (ADX[i] <= 20) and exit_flag:
        sp         = df['Adj Close']
        support    = np.min(np.array(sp[i-15:i]))
        resistance = np.max(np.array(sp[i-15:i]))

        res[i]     = todaysprice - resistance
        sup[i]     = todaysprice - support
        ########## Action of Selling ###########
        if todaysprice >= resistance and percent_profit[i] > 0 and daily_ret_port[i] > 0:
            if portfolio[1] == 1:
                shares_s[i] = 1
                Amt_s[i]    = todaysprice
                stockNo     = 0
                stockVal    = stockNo * todaysprice
                cashVal     = portfolio[0] + Amt_s[i]
            else:
                shares_s[i] = int(portfolio[1] * 0.25)
                Amt_s[i]    = shares_s[i] * todaysprice
                stockNo     = portfolio[1] - shares_s[i]
                stockVal    = stockNo * todaysprice
                cashVal     = portfolio[0] + Amt_s[i]

            portfolio       = [cashVal, stockNo, stockVal]
            buysell1[i]     = -5
            sell[i]         = 1
            rew_s[i]        = sell[i]*100 + 20*percent_profit[i]
            rew_b[i]        = -sell[i]*100 - 20*percent_profit[i]
            print i, "Sell in no trend"

        ########## Action of buying Stocks ##############
        elif todaysprice <= support:
            if portfolio[0] < todaysprice:
                shares_b[i]            = 0
                Amt_b[i]               = 0
                effective_buying_price = (effective_buying_price * portfolio[1] + Amt_b[i])/(portfolio[1] + shares_b[i])
                stockNo                = portfolio[1]
                stockVal               = stockNo * todaysprice
                cashVal                = portfolio[0]
            else:
                shares_b[i]            = int(portfolio[0] * 0.3 / todaysprice)
                Amt_b[i]               = shares_b[i] * todaysprice
                effective_buying_price = (effective_buying_price * portfolio[1] + Amt_b[i])/(portfolio[1] + shares_b[i])
                stockNo                = portfolio[1] + shares_b[i]
                stockVal               = stockNo * todaysprice
                cashVal                = portfolio[0] - Amt_b[i]

            portfolio   = [cashVal, stockNo, stockVal]
            buysell1[i] = 5
            buy[i]      = 1
            rew_b[i]    = buy[i]*100
            rew_s[i]    = -buy[i]*100
            print i, "Buy in no trend"

        ########### Action of Holding Stocks ###########
        else:
            portfolio   = [portfolio[0], portfolio[1], portfolio[1] * todaysprice]
            hold[i]     = 1
            rew_h[i]    = hold[i]*100
            print i, "Hold in no trend"

        ######## Appending calculated factors in a csv file ############
        no_trend_data.append([sup[i], res[i], percent_profit[i], daily_ret_port[i], rew_b[i], rew_s[i], rew_h[i]])

    else:
        portfolio = [portfolio[0], portfolio[1], portfolio[1] * todaysprice]
        hold[i]   = 100
        print i, "HOLD"
    
    ############## Appending all data for Analysis ###################
    data.append([portfolio[0], portfolio[1], get_value(df, i), portfolio[2], net_worth[i], shares_b[i], Amt_b[i], shares_s[i], Amt_s[i], buy[i], sell[i], hold[i]])

############ saving csv files ##################
pyex.save_as(array=data, dest_file_name='data_with_branching_model.csv')
pyex.save_as(array=trend_data, dest_file_name='trending_market_data.csv')
pyex.save_as(array=no_trend_data, dest_file_name='sideways_market_data.csv')

########### Plotting data ###################
plt.subplot(311)
df['Adj Close'].plot()
plt.subplot(312)
plt.plot(buysell)
plt.subplot(313)
plt.plot(net_worth)
plt.show()
