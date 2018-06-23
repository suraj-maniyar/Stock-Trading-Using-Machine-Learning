import pandas as pd
import numpy as np
import math
import csv


path = '/home/suraj/fyp/fyp2/FUNDAMENTAL/fdata/'
company = pd.DataFrame.from_csv("sectorwise_list.csv")
key_stats = pd.DataFrame.from_csv("key_stats.csv")
data = [['TICKER','Pass Score', 'Fail Score','Ratio']]
data_with_IO_Error = []
data_with_Value_Error = []
data_with_Key_Error = []
index=0

Ticker_list = []
with open('EQUITY_LIST_NSE.csv') as csvfile:
   reader = csv.DictReader(csvfile)
   for row in reader:
       ticker = row['SYMBOL']
       Ticker_list.append(ticker)
Ticker_list = list(filter(lambda x: x !='', Ticker_list))


'# Revenue'
def get_revenue_annual(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_IS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Revenue'].replace('-','0')  

def get_revenue_quarterly(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_IS_quarterly_data.csv'    
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Revenue'].replace('-','0')    


    
'# Gross Profit'    
def get_gross_profit_annual(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_IS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Gross Profit'].replace('-','0')    

def get_gross_profit_quarterly(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_IS_quarterly_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Gross Profit'].replace('-','0') 



'# Investing Cash Flow'
def get_investing_cash_flow(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_CF_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Cash Flows From Investing Activities'].replace('-','0')



'# Total Assets'
def get_total_assets(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_BS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Assets'].replace('-','0')
    

    
'# Net Income'
def get_net_income_annual(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_IS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa).reset_index(drop = True)
    return df.loc[24].replace('-','0')    
    
def get_net_income_quarterly(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_IS_quarterly_data.csv'
    df = pd.DataFrame.from_csv(pa).reset_index(drop = True)
    return df.loc[24].replace('-','0')    


    
'# ShareHolder Equity OR Common Stock'    
def get_common_stock_annual(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_BS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Common Stock'].replace('-','0')    

def get_common_stock_quarterly(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_BS_quarterly_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Common Stock'].replace('-','0')    
        

        
'# Return on Equity'        
def get_return_on_equity_annual(ticker):
    common_stock = get_common_stock_annual(ticker).values.astype('float')
    net_income = get_gross_profit_annual(ticker).values.astype('float')
    return np.divide(net_income,common_stock)

def get_return_on_equity_quarterly(ticker):
    common_stock = get_common_stock_quarterly(ticker).values.astype('float')
    net_income = get_gross_profit_quarterly(ticker).values.astype('float')
    return np.divide(net_income,common_stock)



'# Price/Book'
def get_PriceToBook(ticker):
    return key_stats.loc[ticker,'Price/Book']



'# EBITDA'
def get_EBITDA(ticker):
    return key_stats.loc[ticker,'Enterprise Value/EBITDA']



'# Price to Sales'
def get_PricetoSales(ticker):
    return key_stats.loc[ticker,'Price/Sales']



'# Market Capatilization'
def get_MarketCap(ticker):
    return key_stats.loc[ticker,'Market Cap']



'# Gear Ratio'
def get_gear_ratio(ticker):
    debt = key_stats.loc[ticker,'Total Debt']
    market_cap = key_stats.loc[ticker,'Market Cap']
    return np.divide(debt,market_cap) 



'# Cash Flow Operation'    
def get_cash_flow_op_annual(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_CF_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Cash Flow From Operating Activities'].replace('-','0')    

def get_cash_flow_op_quarterly(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_CF_quarterly_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Cash Flow From Operating Activities'].replace('-','0')    



'# Current Assets'
def get_current_assets(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_BS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Current Assets'].replace('-','0')

        
        
'# Current Liabilities'        
def get_current_liabilities(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_BS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Current Liabilities'].replace('-','0')



'# Current Ratio'
def get_current_ratio(ticker):
    current_assets = np.array(get_current_assets(ticker)).astype('float')
    current_liabilities = np.array(get_current_liabilities(ticker)).astype('float')
    ratio = np.divide(current_assets,current_liabilities)
    return ratio    


        
'#Capital Expenditure'    
def get_capital_expenditure_annual(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_CF_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Capital Expenditures'].replace('-','0')    

def get_capital_expenditure_quarterly(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_CF_quarterly_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Capital Expenditures'].replace('-','0')    


    
'#Total Assets'    
def get_total_assets_annual(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_BS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Assets'].replace('-','0')    

def get_total_assets_quarterly(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_BS_quarterly_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Total Assets'].replace('-','0')



'# Asset Turnover Ratio'
def get_asset_turnover_annual(ticker):
    asset = get_total_assets_annual(ticker).values.astype('float')
    revenue = get_revenue_annual(ticker).values.astype('float')    
    return np.divide(asset,revenue)

def get_asset_turnover_quarterly(ticker):
    asset = get_total_assets_quarterly(ticker).values.astype('float')
    revenue = get_revenue_quarterly(ticker).values.astype('float')    
    return np.divide(asset,revenue)



'# Earnings'    
def get_earnings_annual(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_IS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Earnings Before Interest and Taxes'].replace('-','0')    

def get_earnings_quarterly(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_IS_quarterly_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Earnings Before Interest and Taxes'].replace('-','0') 



'# Return on Assets'
def get_return_on_assets(ticker):
    net_income = np.array(get_net_income_annual(ticker)).astype('float')
    total_assets = np.array(get_total_assets(ticker)).astype('float')
    roa = np.divide(net_income,total_assets)
    return roa



'# Long Term Debt'
def get_long_term_debt(ticker):
    pa = path + ticker[0] + '/'+ticker+'/' + ticker+'_BS_annual_data.csv'
    df = pd.DataFrame.from_csv(pa)
    return df.loc['Long Term Debt'].replace('-','0') 


    
'# Debt to Equity'
def get_DE_ratio(ticker):
    return key_stats.loc[ticker,'DE Ratio']



'# PEG ratio'
def get_PEG(ticker):
    return key_stats.loc[ticker,'PEG Ratio']



'#Asset Turnover ratio'
def get_asset_turnover_ratio(ticker):
    revenue = np.array(get_revenue_annual(ticker)).astype('float')
    total_assets = np.array(get_total_assets(ticker)).astype('float')
    TR = np.divide(revenue,total_assets)
    return TR



'# PE Ratio'
def get_PE_ratio(ticker):
    pa = path+'sectorwise_list_use_this.csv'    
    df = pd.DataFrame.from_csv(pa)
    val = df.loc[ticker,'PE']
    if(val == 'NM'):
        val = 0
    return float(val)    



'# EPS'
def get_EPS(ticker):
    pa = path+'sectorwise_list_use_this.csv'    
    df = pd.DataFrame.from_csv(pa)
    val = df.loc[ticker,'EPS']
    if(val == 'NM'):
        val = 0
    return float(val)    


'# DE average'
def get_DE_average(ticker):
    sector = company.loc[ticker,'SECTOR']
    market_cap = []
    DE = []

    for i in range(0,len(Ticker_list)):
        if(company.loc[Ticker_list[i],'SECTOR'] == sector):
            temp1 = get_MarketCap(Ticker_list[i])
            if(math.isnan(temp1)):
                temp1 = 0
            market_cap.append(temp1)
          
            temp2 = get_DE_ratio(Ticker_list[i])
            if(math.isnan(temp2)):
                temp2 = 0
            DE.append(temp2)

    DE = np.array(DE).astype('float')
    market_cap = np.array(market_cap).astype('float')
    summ = np.sum(market_cap)
    weight = np.divide(market_cap,summ)
    multip = np.multiply(weight,DE)
    avg = np.sum(multip)        
    return avg

    
'# PE average'
def get_PE_average(ticker):
    sector = company.loc[ticker,'SECTOR']
    market_cap = []
    PE = []

    for i in range(0,len(Ticker_list)):
        if(company.loc[Ticker_list[i],'SECTOR'] == sector):
            print Ticker_list[i]
            temp1 = get_MarketCap(Ticker_list[i])
            if(math.isnan(temp1)):
                temp1 = 0
            market_cap.append(temp1)
          
            temp2 = get_PE_ratio(Ticker_list[i])
            if(math.isnan(temp2)):
                temp2 = 0
            PE.append(temp2)

    PE = np.array(PE).astype('float')
    market_cap = np.array(market_cap).astype('float')
    summ = np.sum(market_cap)
    weight = np.divide(market_cap,summ)
    multip = np.multiply(weight,PE)
    avg = np.sum(multip)        
    return avg


'# EBITDA average'
def get_EBITDA_average(ticker):
    sector = company.loc[ticker,'SECTOR']
    market_cap = []
    EBITDA = []

    for i in range(0,len(Ticker_list)):
        if(company.loc[Ticker_list[i],'SECTOR'] == sector):
            
            temp1 = get_MarketCap(Ticker_list[i])
            if(math.isnan(temp1)):
                temp1 = 0
            market_cap.append(temp1)
          
            temp2 = get_EBITDA(Ticker_list[i])
            if(math.isnan(temp2)):
                temp2 = 0
            EBITDA.append(temp2)

    EBITDA = np.array(EBITDA).astype('float')
    market_cap = np.array(market_cap).astype('float')
    summ = np.sum(market_cap)
    weight = np.divide(market_cap,summ)
    multip = np.multiply(weight,EBITDA)
    avg = np.sum(multip)        
    return avg


'# Price to Sales average'
def get_Pricetosales_average(ticker):
    sector = company.loc[ticker,'SECTOR']
    market_cap = []
    P2S = []

    for i in range(0,len(Ticker_list)):
        if(company.loc[Ticker_list[i],'SECTOR'] == sector):
            temp1 = get_MarketCap(Ticker_list[i])
            if(math.isnan(temp1)):
                temp1 = 0
            market_cap.append(temp1)
          
            temp2 = get_PricetoSales(Ticker_list[i])
            if(math.isnan(temp2)):
                temp2 = 0
            P2S.append(temp2)

    P2S = np.array(P2S).astype('float')
    market_cap = np.array(market_cap).astype('float')
    summ = np.sum(market_cap)
    weight = np.divide(market_cap,summ)
    multip = np.multiply(weight,P2S)
    avg = np.sum(multip)        
    return avg



'# Asset Turnover Average'
def get_asset_turnover_average(ticker):
    sector = company.loc[ticker,'SECTOR']
    market_cap = []
    atr = []
    
    for i in range(0,len(Ticker_list)):
        if(company.loc[Ticker_list[i],'SECTOR'] == sector): 
            temp1 = get_asset_turnover_ratio(Ticker_list[i])[0]
            if(math.isnan(temp1)):
                temp1 = 0
            atr.append(temp1)
            
            temp2 = get_MarketCap(Ticker_list[i])
            if(math.isnan(temp2)):
                temp2 = 0
            market_cap.append(temp2)
            
    atr = np.array(atr).astype('float')
    market_cap = np.array(market_cap).astype('float')
    summ = np.sum(market_cap)
    weight = np.divide(market_cap,summ)
    multip = np.multiply(weight,atr)
    avg = np.sum(multip)                                       
    return avg


'# Avg revenue growth'
def get_revenue_growth_average(ticker):
  revenue_growth = []
  market_cap = []
   
  try:  
    sector = company.loc[ticker,'SECTOR']   
  except KeyError:
    return np.NaN
  else:    
    
    for i in range(0,len(Ticker_list)):
        if(company.loc[Ticker_list[i],'SECTOR'] == sector):
           temp1 = get_MarketCap(Ticker_list[i])
           if(math.isnan(temp1)):
                temp1 = 0
             
            
           try: 
               revenue = get_revenue_annual(Ticker_list[i])
               revenue = np.array(revenue.values.astype('float'))
               try: 
                  pg = 100.0*(revenue[0] - revenue[1])/revenue[1]          
                  if(math.isnan(pg)):
                    pass
                  else:
                    market_cap.append(temp1)
                    revenue_growth.append(pg)
                    
               except IndexError:
                   pass               
           except IOError:
              pass  
              
    revenue_growth = np.array(revenue_growth).astype('float')
    market_cap = np.array(market_cap).astype('float')
    summ = np.sum(market_cap)
    weight = np.divide(market_cap,summ)
    multip = np.multiply(weight,revenue_growth)
    avg = np.sum(multip)  
    return avg

    
'###################################################################################'



def calculate_score(ticker):
  pass_score= 0
  fail_score = 0   

  revenue =  get_revenue_annual(ticker)
  revenue = np.array(revenue.values.astype('float')) 
  try:
     if( (revenue[0]>=revenue[1]) and (revenue[1]>=revenue[2]) ):
         pass_score = pass_score+1
     else:
         fail_score = fail_score+1       
  except IndexError:
     pass
 

  gross_profit = get_gross_profit_annual(ticker)
  GP = np.array(gross_profit.values.astype('float')) 
  try:
     if( (GP[0]>GP[1]) and (GP[1]>GP[2]) ):
         pass_score = pass_score+1
     else:
         fail_score = fail_score+1       
  except IndexError:
      pass
  
  try: 
     RE = get_return_on_equity_annual(ticker)   
  except ValueError:
      pass
  else:
     try: 
        if(RE[0]>RE[1]):
            pass_score = pass_score+1
        else:
            fail_score = fail_score+1
     except IndexError:
        pass  
     


  PB = get_PriceToBook(ticker)
  if (math.isnan(PB)):
      PB = 0 
  if(PB<1):
      pass_score += 1
  else:
      fail_score += 1
    
  GeaR = get_gear_ratio(ticker) 
    
  if (math.isnan(GeaR)):
      GeaR = 0
  elif(GeaR<1):
      pass_score += 1
  elif(GeaR>1):
      fail_score += 1
    
  CE = get_capital_expenditure_annual(ticker)
  CE = abs(np.array(CE.values.astype('float')))
  OCF = get_cash_flow_op_annual(ticker)
  OCF = np.array(OCF.values.astype('float'))
  
  try:
    if(OCF[0]>CE[0]):
       pass_score += 1
    elif(OCF[0]<CE[0]):
       fail_score += 1
  except IndexError:
     pass   
  
  try:     
     if( (OCF[0]>=OCF[1]) and (OCF[1]>=OCF[2]) ):
         pass_score = pass_score+1
     else:
         fail_score = fail_score+1       
  except IndexError:
     pass  
      
  earnings = get_earnings_annual(ticker)
  earnings = np.array(earnings.values.astype('float'))
  common_stock = get_common_stock_annual(ticker)
  common_stock =  np.array(common_stock.values.astype('float'))

  try:
     EPS = np.divide(earnings,common_stock) 
  except ValueError:
     pass
  else:
     try: 
        temp = EPS[0]/EPS[2]
     except IndexError:
         pass
     else:
        if(math.isnan(temp)):
            temp = 0
        if(temp>0):    
            temp = ( math.pow(temp,1.0/3.0) - 1 )*100
        elif(temp<0):
            temp = -( math.pow(-temp,1.0/3.0) - 1 )*100

        if(temp>5):
            pass_score += 1

  total_assets = np.array(get_total_assets(ticker)).astype('float')
  icf = np.array(get_investing_cash_flow(ticker)).astype('float')
  
  try:
     if((icf[0]<0)):
        if(total_assets[1]<total_assets[0]):
            pass_score += 1
        else:
            fail_score += 1
  except IndexError:
       pass
              

  cur_ratio = get_current_ratio(ticker)
  try:
     if( (cur_ratio[0]>cur_ratio[1]) and (cur_ratio[1]>cur_ratio[2]) ):
         pass_score += 1
     else:
         fail_score += 1
  except IndexError:
     pass
 

  try: 
     roa = get_return_on_assets(ticker) 
  except ValueError:
     pass
  else:
     try:  
        if( (roa[0]>roa[1]) and (roa[1]>roa[2]) ):
            pass_score += 1
        else:
            fail_score += 1    
     except IndexError:
        pass
    
  try:       
     roa1 = get_return_on_assets(ticker)
  except ValueError:
      pass
  else:
      try:
         if(OCF[0] > roa1[0]):
            pass_score += 1
         else:
            fail_score +=1
      except IndexError:
          pass
      
  LTD = get_long_term_debt(ticker)
  LTD = np.array(LTD).astype('float')
  try:
     if((LTD[0] < LTD[1]) and (LTD[1] < LTD[2])):
         pass_score += 1
     else:
         ltd_growth_rate = 100.0*(LTD[0] - LTD[1])/(LTD[1]+1)
         revenue_gr = 0
         try:                 
            revenue = get_revenue_annual(ticker)
            revenue = np.array(revenue.values.astype('float'))            
         except IOError:
            revenue_gr =0  
         try:
            pg1 = 100.0*(revenue[0] - revenue[1])/revenue[1]          
            if(math.isnan(pg1)):
              revenue_gr =0
            else:
              revenue_gr =  pg1 
         except IndexError:
              revenue_gr =0
         if (revenue_gr > ltd_growth_rate):
             pass_score +=1
         else:
             fail_score +=1
  except IndexError:
     pass
 
  try:
      ATR =  get_asset_turnover_ratio(ticker)[0]
      ATR_avg = get_asset_turnover_average(ticker)
  except (ValueError,IndexError):
      pass
  else:
      try:
         if(ATR>ATR_avg):
            pass_score += 1
         else:
            fail_score += 1    
      except IndexError:
         pass 

  try:
     if(get_DE_ratio(ticker) < get_DE_average(ticker) ):
        pass_score += 1
     else:
        fail_score += 1
  except Exception:
      pass 
   
  try:    
   revenue = get_revenue_annual(ticker)
   revenue = np.array(revenue.values.astype('float'))
  except IOError:
       print "IO error.................." 
       
  
  try:
       
       percent_growth = 100.0*(revenue[0] - revenue[1])/revenue[1]          
       if(math.isnan(percent_growth)):
              pass
       else:
              if(percent_growth>get_revenue_growth_average(ticker)):
                  pass_score += 1           
  except IndexError:
              pass
 
  
  ebitda = get_EBITDA(ticker)
  if(math.isnan(ebitda)):
      pass
  else:
      if(ebitda<=10):
          pass_score += 1

  try:        
     asset = get_current_assets(ticker)
     liab = get_current_liabilities(ticker)       
  except IOError:
     pass
  else:
     try:
         asset = np.array(asset).astype('float')
         liab = np.array(liab).astype('float')
     except IndexError:
         pass
     else:
         arr = asset-liab
         if(np.min(arr)>=0):
             pass_score += 1
         
  try:   
    PE = get_PE_ratio(ticker)
    PE_avg = get_PE_average(ticker)
  except(IOError,ValueError,IndexError):
    pass
  else:
    if(PE>PE_avg):
        pass
    elif(PE<PE_avg):
        if(pass_score>fail_score):
            pass_score += 2
      
      
  return pass_score,fail_score




with open('EQUITY_LIST_NSE.csv') as csvfile:
   reader = csv.DictReader(csvfile)
   
   for row in reader:
       ticker = row['SYMBOL']
       index=index+1
       if(ticker == ''):
          pass
       else:
          print str(index)+') ',ticker
          try:      
               p,f = calculate_score(ticker)
               data.append([ticker,p,f,1.0*p/f])
               print "  ",p,f
          except (IOError,ValueError) as e:
               #print e
               print type(e)
               if(type(e) == IOError):
                  print "#####################################" 
                  data_with_IO_Error.append(row) 
               elif(type(e) == ValueError):
                  print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" 
                  data_with_Value_Error.append(row)
               elif(type(e) == KeyError):
                  print "*************************************" 
                  data_with_Key_Error.append(row)
                              


with open ("Fundamental_score.csv","wb") as f:
    writer = csv.writer(f)
    writer.writerows(data)

io = [[]]
for i in range(0,len(data_with_IO_Error)):
    io.append(data_with_IO_Error[i].values())               
with open ("IO_error.csv","wb") as f:
    writer = csv.writer(f)
    writer.writerows(io)

val = [[]]
for i in range(0,len(data_with_Value_Error)):
    val.append(data_with_Value_Error[i].values())
with open ("Value_error.csv","wb") as f:
    writer = csv.writer(f)
    writer.writerows(val)

key = [[]]
for i in range(0,len(data_with_Key_Error)):
    key.append(data_with_Key_Error[i].values())
with open ("Key_error.csv","wb") as f:
    writer = csv.writer(f)
    writer.writerows(key)


          
          
          