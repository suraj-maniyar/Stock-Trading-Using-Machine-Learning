from selenium import webdriver
import codecs
from selenium.webdriver.common.by import By
import os
import csv
import time


with open('IO_error.csv') as csvfile:
   reader = csv.DictReader(csvfile)
   for row in reader:
     path = '/home/suraj/fyp/fyp2/FUNDAMENTAL/fdata/'         
     path_annual = path + row['SYMBOL']+'/'+row['SYMBOL']+'_BS_annual_data.csv'
     path_quarterly = path + row['SYMBOL']+'/'+row['SYMBOL']+'_BS_quarterly_data.csv'         
        
     ticker_name = row['SYMBOL'] + '.NS'
     print ticker_name
     if (not ( os.path.isfile(path_annual) and os.path.isfile(path_quarterly) )):  
       firefox_profile = webdriver.FirefoxProfile()
       firefox_profile.set_preference('permissions.default.image', 2)
       firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
       browser = webdriver.Firefox(firefox_profile=firefox_profile)
        
       url = 'https://finance.yahoo.com/quote/' +ticker_name +'/balance-sheet?p=' +ticker_name
       browser.get(url)
       data = []
       if(not os.path.isdir(path+row['SYMBOL'])):
         os.mkdir(path + row['SYMBOL'])
       else:
         if(not (os.path.isfile(path_annual))):
           html_source_annual = browser.page_source
           #element_bs = browser.find_element(By.XPATH, '//*[@id="main-0-Quote-Proxy"]/section/div[2]/section/div/section/div[2]/button[2]')
           #element_bs = browser.find_element(By.XPATH, '//*[@id="quote-leaf-comp"]/section/div[1]/div[1]/a[1]/div/span')
           #element_bs.click()  
        
           #html_source_annual = browser.page_source
           fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_annual')
         
           with codecs.open(fullpath, "w", encoding="utf-8") as f:
              f.write(html_source_annual)
           print "annual saved"
           #element_A = browser.find_element_by_xpath('//*[@id="main-0-Quote-Proxy"]/section/div[2]/section/div/section/div[4]/table/tbody')
           element_A = browser.find_element_by_xpath('//*[@id="quote-leaf-comp"]/section/div[3]/table/tbody')
           fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_annual_table')
           with codecs.open(fullpath, "w", encoding="utf-8") as f:
                 f.write(element_A.text)
        
           with open(fullpath, 'r') as data:
               plaintext_A = data.read()
           plaintext_A = plaintext_A.replace(',', '')
           plaintext_A = plaintext_A.replace(' 0', ',0')
           plaintext_A = plaintext_A.replace(' 1', ',1')
           plaintext_A = plaintext_A.replace(' 2', ',2')
           plaintext_A = plaintext_A.replace(' 3', ',3')
           plaintext_A = plaintext_A.replace(' 4', ',4')
           plaintext_A = plaintext_A.replace(' 5', ',5')
           plaintext_A = plaintext_A.replace(' 6', ',6')
           plaintext_A = plaintext_A.replace(' 7', ',7')
           plaintext_A = plaintext_A.replace(' 8', ',8')
           plaintext_A = plaintext_A.replace(' 9', ',9')
           plaintext_A = plaintext_A.replace(' -', ',-')
           fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_annual_data.csv')
           file_A = open(fullpath,"w")
           file_A.write(plaintext_A)
           file_A.close()
           print "annual data written to table"
         else:
           print row['SYMBOL']+'_BS_annual_data.csv'+' already exists!'  
        
         if(not (os.path.isfile(path_quarterly))):
           #el = browser.find_element(By.XPATH, '//*[@id="main-0-Quote-Proxy"]/section/div[2]/section/div/section/div[2]/div[2]/button')
           el = browser.find_element(By.XPATH, '//*[@id="quote-leaf-comp"]/section/div[1]/div[2]/button/div/span')
           el.click()
           html_source_quarterly = browser.page_source

           fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_quarterly')
           with codecs.open(fullpath, "w", encoding="utf-8") as f:
               f.write(html_source_quarterly)
           print "quarterly saved"
           #element_Q = browser.find_element_by_xpath('//*[@id="main-0-Quote-Proxy"]/section/div[2]/section/div/section/div[4]/table/tbody')
           element_Q = browser.find_element_by_xpath('//*[@id="quote-leaf-comp"]/section/div[3]/table/tbody')
           fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_quarterly_table')
           with codecs.open(fullpath, "w", encoding="utf-8") as f:
               f.write(element_Q.text)

           with open(fullpath, 'r') as data:
               plaintext_Q = data.read()
           plaintext_Q = plaintext_Q.replace(',', '')
           plaintext_Q = plaintext_Q.replace(' 0', ',0')
           plaintext_Q = plaintext_Q.replace(' 1', ',1')
           plaintext_Q = plaintext_Q.replace(' 2', ',2')
           plaintext_Q = plaintext_Q.replace(' 3', ',3')
           plaintext_Q = plaintext_Q.replace(' 4', ',4')
           plaintext_Q = plaintext_Q.replace(' 5', ',5')
           plaintext_Q = plaintext_Q.replace(' 6', ',6')
           plaintext_Q = plaintext_Q.replace(' 7', ',7')
           plaintext_Q = plaintext_Q.replace(' 8', ',8')
           plaintext_Q = plaintext_Q.replace(' 9', ',9')
           plaintext_Q = plaintext_Q.replace(' -', ',-')
           fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_quarterly_data.csv')
           file_Q = open(fullpath,"w")
           file_Q.write(plaintext_Q)
           file_Q.close()
           print "quarterly data written to table"
         else:
           print row['SYMBOL']+'_BS_quarterly_data.csv'+' already exists!'         
        
         fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_quarterly')
         os.remove(fullpath)
         fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_annual')
         os.remove(fullpath)
         fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_quarterly_table')
         os.remove(fullpath)
         fullpath=os.path.join(path, row['SYMBOL'], row['SYMBOL'] +'_BS_annual_table')
         os.remove(fullpath)
         browser.quit()
         time.sleep(2)