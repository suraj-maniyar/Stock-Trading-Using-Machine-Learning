                              Source Code No - 6 

########### Source code for Training Neural Network #######################

'''---------Libraries---------'''
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import model_from_json
'''---------Libraries---------'''



############Reading DataFrame##################   
df1 = pd.read_csv('data_intrend_HCL_Tech.csv')
df2 = pd.read_csv('data_notrend_HCL_Tech.csv')
###############################################

############## Functions for saving and loading ML model#####################
def newModel():
    model1 = Sequential()
    model1.add(Dense(100, input_dim=input_size1, init='lecun_uniform', activation='relu'))
    model1.add(Dense(50, init='lecun_uniform', activation='relu'))
    model1.add(Dense(output_size1, init='lecun_uniform'))

    model2 = Sequential()
    model2.add(Dense(100, input_dim=input_size2, init='lecun_uniform', activation='relu'))
    model2.add(Dense(50, init='lecun_uniform', activation='relu'))
    model2.add(Dense(output_size2, init='lecun_uniform'))
    return model1,model2

########### Function for saving ML model ##################################    
def saveModel(model,k):
    model_json = model.to_json()
    if(k==1):
       with open("model1.json", "w") as json_file:
          json_file.write(model_json)
       model.save_weights("model1.h5")
       print("Saved model-1 to disk")
    elif(k==2):
       with open("model2.json", "w") as json_file:
          json_file.write(model_json)
       model.save_weights("model2.h5")
       print("Saved model-2 to disk")

############Function for loading ML model ##############################
def loadModel():
    json_file         = open('model1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model1            = model_from_json(loaded_model_json)
    model1.load_weights("model1.h5")
    print("Loaded model-1 from disk")
    
    json_file         = open('model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model2            = model_from_json(loaded_model_json)
    model2.load_weights("model2.h5")
    print("Loaded model-2 from disk")
    return model1,model2

############################################################

############ Variable Initialization ###########
input_size1  = 10
input_size2  = 4
output_size1 = 7
output_size2 = 3
start        = 20
len_test     = 20
len_train1   = len(df1)-len_test-start
len_train2   = len(df2)-len_test-start
#################################################

############## Reading all columns of a csv #######
td1             = df1['todaysprice']
td1_ma          = td1.rolling(window=20,center=False).mean() 
Ub              = df1['Boll_UP']
Lb              = df1['Boll_LOWER']
cci             = df1['CCI']
rsi             = df1['RSI']
trend_dir_adl   = df1['trend_dir_adl']
trend_dir_data  = df1['trend_dir_data']
action1         = df1['buysell']
macd            = df1['MACD']
pp              = df1['Percent_Profit']
drp             = df1['Daily_Return_Portfolio']
temp            = macd[len(df1)-1]
macd_shifted    = [temp]*len(df1)
###################################################

############## Data Preprocessing ###################

for i in range(1,len(df1)):
    macd_shifted[i] = macd.get_value(i-1,'MACD')
 
macd_shifted        = np.transpose(macd_shifted) 
td1                 = td1[start:len(df1)]
td1_ma              = td1_ma[start:len(df1)]
Ub                  = Ub[start:len(df1)]
Lb                  = Lb[start:len(df1)]
cci                 = cci[start:len(df1)]
rsi                 = rsi[start:len(df1)]
trend_dir_data      = trend_dir_data[start:len(df1)]
trend_dir_adl       = trend_dir_adl[start:len(df1)]
action1             = action1[start:len(df1)]
macd                = macd[start:len(df1)]
macd_shifted        = macd_shifted[start:len(df1)]
pp                  = pp[start:len(df1)]
drp                 = drp[start:len(df1)]

td1                 = td1.reset_index(drop=True)
td1_ma              = td1_ma.reset_index(drop=True)
Ub                  = Ub.reset_index(drop=True)
Lb                  = Lb.reset_index(drop=True)
rsi                 = rsi.reset_index(drop=True)
cci                 = cci.reset_index(drop=True)
macd                = macd.reset_index(drop=True)
action1             = action1.reset_index(drop=True) 
trend_dir_data      = trend_dir_data.reset_index(drop=True) 
trend_dir_adl       = trend_dir_adl.reset_index(drop=True) 
pp                  = pp.reset_index(drop=True)
drp                 = drp.reset_index(drop=True)

td2                 = df2['todaysprice']
support             = df2['Support']
resistance          = df2['Resistance']
action2             = df2['Action']
pp2                 = df2['Percent_Profit']
drp2                = df2['Daily_Return_Portfolio']

td2                 = td2[start:len(df2)]
support             = support[start:len(df2)]
resistance          = resistance[start:len(df2)]
action2             = action2[start:len(df2)]  
pp2                 = pp2[start:len(df2)]
drp2                = drp2[start:len(df2)]
td2                 = td2.reset_index(drop=True)

support             = support.reset_index(drop=True)
resistance          = resistance.reset_index(drop=True)
action2             = action2.reset_index(drop=True)
pp2                 = pp2.reset_index(drop=True)
drp2                = drp2.reset_index(drop=True)
#####################################################################

########## Generating input test vectors ###########################
ip1 = [[td1_ma-Ub],[td1_ma-Lb],[macd],[macd_shifted],[cci],[rsi],[trend_dir_adl],[trend_dir_data],[pp],[drp]]
ip1 = np.array(ip1)
ip1 = np.transpose(ip1)

ip2 = [[td2-support],[td2-resistance],[pp2],[drp2]]
ip2 = np.array(ip2)
ip2 = np.transpose(ip2)

op_train1 = [[[1,2,3]]]*len_train1
op_train1 = np.array(op_train1)
op_train2 = [[[1,2,3]]]*len_train2
op_train2 = np.array(op_train2)

op_test1 = [[[1,2,3]]]*len_test
op_test1 = np.array(op_test1)
op_test1 = op_test1.astype('float32')

op_test2 = [[[1,2,3]]]*len_test
op_test2 = np.array(op_test2)
op_test2 = op_test2.astype('float32')


for i in range(0,len_train1):
    if(action1[i] == 0):
        op_train1[i] = [1,0,0] # Buy 
    elif(action1[i] == 1):
        op_train1[i] = [0,1,0] # Sell
    elif(action1[i] == 2):
        op_train1[i] = [0,0,1] # Hold
        
for i in range(0,len_train2):
    if(action2[i] == 0):
        op_train2[i] = [1,0,0] # Buy
    elif(action2[i] == 1):
        op_train2[i] = [0,1,0] # Sell
    elif(action2[i] == 2):
        op_train2[i] = [0,0,1] # Hold

####################################################################

############# Loading new model ################################## 
model1,model2 = newModel()
rms           = RMSprop()
model1.compile(loss='mse', optimizer=rms)
model2.compile(loss='mse', optimizer=rms)

################ Training new model #############################

for i in range(0,len_train1):
    print i
    model1.fit(ip1[i], op_train1[i], batch_size=1, nb_epoch=1, verbose=0)    
print "training-1 complete"


for i in range(0,len_train2):
    print i
    model2.fit(ip2[i], op_train2[i], batch_size=1, nb_epoch=1, verbose=0)    
print "training-2 complete"

################ Saving new model to local filesystem ###############
saveModel(model1,1)
saveModel(model2,2)

#####################################################################

################## Testing for validating Training #################
print "Testing"    
for j in range(len_train1,len_train1+len_test):
    op_test1[j-len_train1,0]       = model1.predict(ip1[j],batch_size=1)    
    index                          = np.argmax(op_test1[j-len_train1,0])
    op_test1[j-len_train1,0]       = [0,0,0,0,0,0,0]
    op_test1[j-len_train1,0,index] = 1
 

output1 = [-1]*len_test
output1 = np.array(output1)

for i in range(0,len_test):
    if(op_test1[i,0,0] == 1):
        output1[i] = 0
    elif(op_test1[i,0,1] == 1):
        output1[i] = 1
    elif(op_test1[i,0,2] == 1):
        output1[i] = 2
        
original_op1 = action1
original_op1 = np.array(original_op1)
original_op1 = original_op1[len_train1:]

print "output1"
print output1
print "original_output1"
print original_op1        


for j in range(len_train2,len_train2+len_test):
    op_test2[j-len_train2,0]       = model2.predict(ip2[j],batch_size=1)    
    index                          = np.argmax(op_test2[j-len_train2,0])
    op_test2[j-len_train2,0]       = [0,0,0]
    op_test2[j-len_train2,0,index] = 1


output2 = [-1]*len_test
output2 = np.array(output2)

for i in range(0,len_test):
    output2[i] = np.argmax(op_test2[i][0])
 
original_op2   = action2
original_op2   = np.array(original_op2)
original_op2   = original_op2[len_train2:]


print "output2"
print output2
print "original_output2"
print original_op2        

##########################################################################
