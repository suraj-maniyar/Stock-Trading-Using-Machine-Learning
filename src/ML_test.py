                              Source Code No - 7  

########### Source code for Testing ML Model ##############  
                 
'''---------Libraries---------'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyexcel
import math as ma
from keras.models import model_from_json
'''---------Libraries---------'''

########## Reading a dataFrame for testing ################

df = pd.read_csv('data/IT/Infosys.csv')

##########################################################

############# Variable Initialization #####################
data_ml_test     = [['Cash','No. of stocks','Action',
                 'todaysprice','net worth','output']] 
start            = 1000
data_width       = 10
moving_avg_width = 5 
portfolio        = [10000,0,0]
len_test         = len(df) - start
################################################################

################ Loading ML model from local Filesystem ########
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)
model1.load_weights("model1.h5")
print("Loaded model-1 from disk")

json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)
model2.load_weights("model2.h5")
print("Loaded model-2 from disk")  

################################################################   

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

ADX , DX , pdi , mdi = get_directional_indicators(df)

################## Data  Preprocessing ################################################
sp           = df['Adj Close']
td           = df['Adj Close']
td_ma        = td.rolling(window=20,center=False).mean()    
sp_ma        = sp.rolling(window=20,center=False).mean()    

Ub, Lb       = get_bollinger_bands(df)
cci          = get_cci(df)
rsi          = get_RSI(df)
adl          = pd.DataFrame(get_money_flow_multiplier(df))
macd         = get_MACD(df)
macd_shifted = [macd[len(df)-1]]*len(df)

for i in range(1,len(df)):
  macd_shifted[i] = macd[i-1]

Ub           = np.array(Ub)
Lb           = np.array(Lb)
td           = np.array(td)
td_ma        = np.array(td_ma)
macd         = np.array(macd)
rsi          = np.array(rsi)
cci          = np.array(cci)
macd_shifted = np.array(macd_shifted)

trend_dir_adl_arr = [0]*len(df)
trend_dir_data_arr = [0]*len(df)

for i in range(start,len(df)):
     print i," / "+str(len(df))
     trend_dir_adl         = get_trend_direction(adl[(i-moving_avg_width-data_width):i])
     trend_dir_data        = get_trend_direction(sp[(i-moving_avg_width-data_width):i]) 
     trend_dir_adl_arr[i]  = trend_dir_adl
     trend_dir_data_arr[i] = trend_dir_data
    

portfolio[2] = portfolio[1]*get_value(df,start-1)
net_worth    = [portfolio[0]+portfolio[2]]*len(df)
exit_flag    = 1

trend_dir_data_arr = np.array(trend_dir_data_arr)
trend_dir_adl_arr  = np.array(trend_dir_adl_arr)
td_ma              = np.array(td_ma)   
td                 = np.array(td)
Ub                 = np.array(Ub)
Lb                 = np.array(Lb)
cci                = np.array(cci)
rsi                = np.array(rsi)
macd               = np.array(macd)
macd_shifted       = np.array(macd_shifted)

Ub                 = Ub[start:start+len_test]
Lb                 = Lb[start:start+len_test]
td                 = td[start:start+len_test]
td_ma              = td_ma[start:start+len_test]
cci                = cci[start:start+len_test]
rsi                = rsi[start:start+len_test]
macd               = macd[start:start+len_test]
macd_shifted       = macd_shifted[start:start+len_test]

ip1                = [ [td_ma-Ub], [td_ma-Lb] ,[macd],
                       [macd_shifted],[cci], [rsi], 
                       [trend_dir_adl_arr], [trend_dir_data_arr] ]
ip1                = np.array(ip1)
ip1                = np.transpose(ip1)
op_test1           = [[[1,2,3]]]*len(df)
sup                = np.min(np.array(sp[start-16:start-1]))
res                = np.max(np.array(sp[start-16:start-1]))

ip2                = [[[]]]*len(df)
ip2[0][0]          = [[[sp[start-1]-sup , sp[start-1]-res]]]

for i in range(0,len(df)): 
   print i," / "+str(len(df)) 
   
   if(i>15):
     support    = np.min(np.array(sp[i-15:i]))
     resistance = np.max(np.array(sp[i-15:i]))
   else:
     support    = 0
     resistance = 0  
   
   ip2[i][0]    = [sp[i] - support, sp[i] - resistance]
   
ip2             = np.array(ip2)

op_test2        = [[[1,2,3]]]*len(df)
action_arr_it   = [0]*len(df)
action_arr_nt   = [0]*len(df)
pp              = [0]*(len(df))
daily_ret_port  = [0]*len(df)
ebp             = get_value(df,start-1)
Amt_b           = [0]*len(df)
Amt_s           = [0]*len(df)
shares_b        = [0]*len(df)
shares_s        = [0]*len(df)
net_worth       = [portfolio[0]+portfolio[2]]*len(df)
portfolio[2]    = portfolio[0]+portfolio[1]*get_value(df,start-1)

####################################################################################

##################### Iterating over Dataframe #####################################
for i in range(start, len(df)):
    action = -1
    print i    
    todaysprice       = get_value(df,i)
    net_worth[i]      = (portfolio[0] + portfolio[1]*todaysprice)
    daily_ret_port[i] = ((net_worth[i] - net_worth[i-1])/net_worth[i-1])*100
    pp[i]             = (((todaysprice-ebp)/ebp)*100)
    
    if((ADX[i]>25) and exit_flag):
        a             = ip1[i]        
        a             = np.append(a,[[pp[i]]],1)
        a             = np.append(a,[[daily_ret_port[i]]],1)        
        op_test1[i]   = model1.predict(a,batch_size=1)
        action        = np.argmax(op_test1[i])
        val0          = op_test1[i][0][0]
        val1          = op_test1[i][0][1]
        val2          = op_test1[i][0][2]
        maximum       = np.max([val0,val1,val2])
        factor        = 1.0*maximum/(val0+val1+val2)
       
        if(action == 0): # Buy 
               action_arr_it[i] = 20*factor
               shares_b[i]      = int(portfolio[0] * factor / todaysprice)
               Amt_b[i]         = shares_b[i] * todaysprice
               ebp              = (ebp*portfolio[1] + Amt_b[i])/(portfolio[1] + shares_b[i])
               print ebp
               stockNo          = portfolio[1] + shares_b[i]
               stockVal         = stockNo * todaysprice
               cashVal          = portfolio[0] - Amt_b[i]
               portfolio        = [cashVal, stockNo, stockVal]
               print "Buy in trend"
               
        elif(action == 1): # Sell
               action_arr_it[i] = -20*factor
               if portfolio[1] == 1:
                 shares_s[i]    = 1
                 Amt_s[i]       = todaysprice
                 stockNo        = 0
                 stockVal       = stockNo * todaysprice
                 cashVal        = portfolio[0] + Amt_s[i]
               else:
                 shares_s[i]    = int(portfolio[1] * factor)
                 Amt_s[i]       = shares_s[i] * todaysprice
                 stockNo        = portfolio[1] - shares_s[i]
                 stockVal       = stockNo * todaysprice
                 cashVal        = portfolio[0] + Amt_s[i]
               print "Sell in trend"
               
        elif(action == 2): # Hold
               action_arr_it[i] = 0
               print "Hold in trend"
    
        portfolio = [cashVal, stockNo, stockVal]
        data_ml_test.append([portfolio[0], portfolio[1], action,
                             todaysprice, net_worth[i], op_test1[i]]) 
        
        
    elif ((ADX[i]<=25) and exit_flag):

        support     = np.min(np.array(sp[i-15:i]))
        resistance  = np.max(np.array(sp[i-15:i]))
        b           = ip2[i]        
        b           = np.append(b,[[pp[i]]],1)
        b           = np.append(b,[[daily_ret_port[i]]],1)   
        op_test2[i] = model2.predict(b,batch_size=1)
        action      = np.argmax(op_test2[i])
        val0        = op_test2[i][0][0]
        val1        = op_test2[i][0][1]
        val2        = op_test2[i][0][2]
        maximum     = np.max([val0,val1,val2])
        factor      = 1.0*maximum/(val0+val1+val2)
        
        if(action == 0):
            print "Buy in no trend"
            shares_b[i]         = int(portfolio[0] * factor / todaysprice)
            Amt_b[i]            = shares_b[i] * todaysprice
            ebp                 = (ebp * portfolio[1] + Amt_b[i]) / (portfolio[1] + shares_b[i])
            stockNo             = portfolio[1] + shares_b[i]
            stockVal            = stockNo * todaysprice
            cashVal             = portfolio[0] - Amt_b[i]
            portfolio           = [cashVal, stockNo, stockVal]
            action_arr_nt[i]    = 20*factor
        
        elif(action == 1):
            print "Sell in no trend"
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
            portfolio           = [cashVal, stockNo, stockVal]
            action_arr_nt[i]    = -20*factor
            
        elif(action == 2):
            portfolio           = [portfolio[0], portfolio[1], portfolio[1] * todaysprice]        
            print "Hold in no trend"
            action_arr_nt[i]    = 0 
         
        data_ml_test.append([portfolio[0], portfolio[1], action, todaysprice, net_worth[i], op_test2[i]]) 
       
    else:
        action                  = 2
        portfolio               = [portfolio[0], portfolio[1], portfolio[1] * todaysprice]
        data_ml_test.append([portfolio[0], portfolio[1], action, todaysprice, net_worth[i] ]) 
       
    net_worth[i]=(portfolio[0]+portfolio[1]*todaysprice)
    
#######################################################################################    

######################    Saving Data for Analysis ####################################
print "Saving data"
pyexcel.save_as(array = data_ml_test, dest_file_name = 'data_ML_test.csv')
print "data SAVED!"

######################  Plotting data #################################################
price = np.array(df['Adj Close'])
price = price[start:len(df)]
plt.subplot(211)
plt.plot(price)
plt.subplot(212)
net_worth = net_worth[start:len(df)]
plt.plot(net_worth)
plt.show()

#######################################################################################
