# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:22:29 2020

@author: Ramin Madarshahian
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from datetime import datetime,timedelta
from keras.models import Sequential
from keras.layers import Dense,Dropout
import robin_stocks as r
from sklearn import preprocessing
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from keras.regularizers import l2,l1,l1_l2
from keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import max_norm

#%%
f = open("..\private\credential_pass.txt", 'r')
x = f.readlines()
f.close()
f = open("..\private\credential_user.txt", 'r')
y = f.readlines()
f.close()
r.login(y[0],x[0])
#%%
ticker_list = ['AMZN','FB', 'NVDA','MSFT','SHOP','NFLX','DAL','KKR', 'BIP', 'TTWO', 'ATVI', 'OKTA', 'NOBL', 'TGP', 'NTES', 'ABBV', 'SAP', \
               'LRCX', 'XOM', 'BEP', 'TWTR', 'CNQ', 'RYAAY', 'BBD', 'WPM', 'CODX', 'GOOGL', \
               'SE', 'ZBH', 'TD', 'SWK', 'BB', 'AME', 'RNG', 'PBA', 'FCAU', 'WDC', 'TFX',\
               'SPXL', 'SQ', 'FIS', 'ALXN', 'STZ', 'RTX', 'DHR', 'BSX', 'TJX', 'V', \
               'TM', 'JETS', 'TOT', 'TSM', 'WCN', 'INSW', 'ALNY', 'TNK', 'EVBG', \
               'MCHP', 'FAS','DHI', 'MCD', 'T', 'ARE', \
               'SABR', 'KO', 'WIX', 'NDSN', 'TSN', 'YNDX', 'LH', 'HMC', 'IAC',\
               'LMT', 'TMUS', 'WMT', 'PKX', 'SQQQ', 'MRO', 'LOW', 'PBR', 'GE', 'TPR',\
               'KEYS', 'PLAY', 'GILD', 'CNK', 'VLO', 'GDDY', 'MRK','LYV', 'VALE', \
               'LHX', 'MAS', 'JNUG', 'TQQQ', 'RGS', 'MU', 'ENB', 'CRH', 'TLK', 'INTU', 'BX', \
               'AEE', 'GM', 'BAP', 'EPAM', 'GT', 'ERI', 'NOC', 'VXRT', 'CPRT', 'EPD', 'LDOS',\
               'AEM', 'O', 'HIG', 'BAH', 'AZN', 'MSI', 'DAL', 'DXCM', 'ETN', 'ROST', 'GMAB', \
               'EOG', 'J', 'LTC', 'SHG', 'DD', 'NICE', 'IBN', 'MG', 'FMS', 'SU', 'MRVL', \
               'CSGP', 'AMC', 'SAN', 'LEN', 'ADM', 'CLX', 'ANTM', 'TEAM', 'SH', 'BAM', 'LBRDA',\
               'PXD', 'AMRN', 'BA', 'JPM', 'BPOP', 'LUV', 'NXPI', 'SBAC', 'INO', 'ICE', 'APTV', \
               'SVC', 'AMD', 'PUK', 'SPXS', 'ON', 'FE', 'SWDBY', 'ATUS', 'BMRN', 'CVS', 'TSLA', \
               'MUR', 'BURL', 'PLD', 'FMC', 'EADSY', 'NCLH', 'KEP', 'EXAS', 'AMP',\
               'EVRI', 'FOSL', 'CHL', 'YUMC', 'DVA', 'ROKU', 'VRTX', 'HUM', 'WMB', 'NOW', \
               'IQV', 'DG', 'COP', 'NBIX', 'STWD', 'INVH', 'VIIX', 'NKE', 'MDT', 'M', \
               'GOL', 'BNS', 'NPSNY', 'TTD', 'UNH', 'CRM', 'GD', 'WIFI', 'C', 'CTLT', 'SNPS', \
               'HSC', 'BAX', 'BABA', 'BGNE', 'NGG', 'AJG', 'SKM', 'BP','SNE', 'SUI', \
               'MOMO', 'CI', 'VOXX','CXO', 'BLK', 'PYPL', 'APPN', 'LULU', 'HON', 'WY',\
               'FMX', 'SBUX', 'TAL', 'WFC', 'CPRI', 'GIB', 'AMT', 'CERN', 'BIDU', 'LNG', 'ZTO', \
               'TECH', 'QSR', 'FCX','MAR', 'FTAI', 'SPY', 'FISV','PWR', 'MGA', \
               'HCA', 'SPLK', 'EQIX', 'SLB', 'SVXY', 'ACN','EXC', 'NYMT', 'CHU', 'OXY',\
               'VOO', 'TECL', 'EDU', 'KMX', 'PM', 'CNC', 'DIS', 'BRK-B', 'MA', 'WAB', 'ADBE', 'AAPL',\
               'DRI', 'AEP', 'FICO', 'ADI', 'AAL', 'TNA', 'TWLO', 'VSLR', 'LB', 'MET', 'LVS', 'BPMC',\
               'PSX', 'SYF', 'JD', 'MPC', 'MDLZ', 'SFTBY', 'SWN','USO', 'AMAT', 'TRU', 'SMFG', \
               'IMGN', 'MGM','SRPT', 'GPN', 'TAK', 'ABT', 'AVGO', 'PEJ', 'VOD', 'SSNC',\
               'HDB', 'IGV', 'CMCSA', 'COF', 'TMO', 'GOOG']

#%% load data:,
#end_date=str(datetime.now()- timedelta(days=0))[:10]
end_date='2020-05-26'
start_date="2017-11-01"
response_end_date='2020-05-27'
response_start_date="2017-11-02"

def raw_data_generator(ticker_list = ticker_list,start_date = start_date, end_date = end_date):
    raw_data = dict()
    for stock in ticker_list:
        time.sleep(0.001)
        h=pdr.get_data_yahoo(stock, start=start_date, end=end_date,progress=False)['Adj Close'].values
        raw_data.update({stock:h[:-1]})  
    return raw_data
#%%
raw_data_all = raw_data_generator(ticker_list = ticker_list,\
                              start_date = start_date, end_date = end_date)
#%% Last price for prediction:
def last_price_generator(ticker_list = ticker_list):
    last_price = dict()
    for symbol in ticker_list:
        if '-' in symbol:
            symbol = symbol.replace("-", ".")
        last_price.update({symbol:float(r.get_latest_price(symbol)[0])})
    return last_price
#%% Last price for prediction:

def anydate_price_generator(ticker_list = ticker_list,date = '2020-05-12',enddate = '2020-05-13'):
    last_price = dict()
    for symbol in ticker_list:
        last_price.update({symbol:pdr.get_data_yahoo(symbol, start=date, end=enddate,progress=False)['Adj Close'].values[0]})
    return last_price
#%%
last_price_all =  last_price_generator(ticker_list = ticker_list)
#%%
def stock_status(Ticker = "AMZN",raw_data=raw_data_all.copy(),\
                 last_price=last_price_all.copy(),epochs=100, batch_size=30,verbose = 0,plot=True):
    response = pdr.get_data_yahoo(Ticker, start=response_start_date, end=response_end_date,\
                                      progress=False)['Adj Close'].values[:-1]
    if Ticker in raw_data.keys():
        print("condition is ",response[-2]==raw_data[Ticker][-1])
        if response[-2]!=raw_data[Ticker][-1]:
            response = response[:-1]
            print("condition is ",response[-2]==raw_data[Ticker][-1])
        
        
    if Ticker not in raw_data.keys():
        h=pdr.get_data_yahoo(Ticker, start=start_date, end=end_date,progress=False)['Adj Close'].values
        raw_data.update({Ticker:h[:-1]}) 
                                                    #we ignore the last sample
        if len(response) < len(raw_data[list(raw_data.keys())[0]]):
            for item in raw_data.items():
                raw_data.update({item[0]:item[1][-len(response):]})
    Input_data = pd.DataFrame.from_dict(raw_data)
    response_std = np.diff(response).std()
    #scale input
    scaler_x = preprocessing.StandardScaler().fit(Input_data)
    Input_data_scaled = scaler_x.transform(Input_data)
    #model
    model = Sequential()
    model.add(Dense(128, input_dim=len(Input_data.T), activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
#    model.add(Dense(16, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001), bias_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu', kernel_initializer='lecun_normal',kernel_regularizer = l1_l2(0.001,0.001),activity_regularizer=l1_l2(0.001,0.001)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='Nadam',metrics=['mae', 'mse'])
#    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10000)
    history = model.fit(Input_data_scaled, response,validation_split=0.4, epochs=epochs, \
              batch_size=batch_size,shuffle = True,verbose = verbose)#, callbacks=[es])
    
    hist = pd.DataFrame(history.history)
    print(f"\n{Ticker} done! with mae of {hist['val_mae'].values[-1]:.4}")
    #Prediction
    if Ticker.replace('-','.') in last_price.keys():
        last_used_price = last_price[Ticker.replace('-','.')]#last price
    else:
        last_used_price = float(r.get_latest_price(Ticker.replace('-','.'))[0])
        last_price[Ticker] = last_used_price
    x_pre = scaler_x.transform(np.array(list(last_price.values())).reshape(1,-1))
    predictions = model.predict(x_pre)
    diff = last_used_price - predictions
    status = str()
    if diff > 3 *response_std:
        status = "Strong sell"
    elif diff > 2*response_std:
        status = "Sell"
    elif diff > -2*response_std and diff < 2*response_std:
        status = "Neutral"
    elif diff < -2*response_std and diff > -3*response_std:
        status = "Buy"
    else:
        status = "Strong buy"
    print(f"{status} with predicted price of {np.round(predictions[0][0],2):.4} and last price of {last_used_price}\n")
    model.save(f'Models\{Ticker}_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model
    # summarize history for loss
    if plot:
        plt.close()
        plt.plot(np.sqrt(history.history['mse']))
        plt.plot(np.sqrt(history.history['val_mse']))
        RMSE = np.round(np.sqrt(hist["val_mse"].values[-1]),2)
        plt.title(f'model RMSE for {Ticker}-ended:{RMSE}')
        plt.ylabel('RMSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig(f'Models\{Ticker}.png')
    return status,predictions[0][0],last_used_price,np.sqrt(hist['val_mse'].values[-1])
#%%
#status,predictions,o_last = stock_status(Ticker = 'AMZN',raw_data=raw_data_all.copy(),\
#                 last_price=last_price_all.copy(),epochs=10000, batch_size=128)

My_holdings = ticker_list.copy()
My_holdings = ['AEP', 'FICO', 'ADI', 'AAL', 'TNA', 'TWLO', 'VSLR', 'LB', 'MET', 'LVS', 'BPMC',\
               'PSX', 'SYF', 'JD', 'MPC', 'MDLZ', 'SFTBY', 'SWN','USO', 'AMAT', 'TRU', 'SMFG', \
               'IMGN', 'MGM','SRPT', 'GPN', 'TAK', 'ABT', 'AVGO', 'PEJ', 'VOD', 'SSNC',\
               'HDB', 'IGV', 'CMCSA', 'COF', 'TMO', 'GOOG']
#%%
STATUS = {}
import csv
with open(f"raw_model_{str(datetime.now()- timedelta(days=0))[:10]}.csv",'w') as csv_file:
    csvwriter = csv.writer(csv_file,lineterminator = '\n') 
    csvwriter.writerow(['ticker','status','Predicted','Last price','RMSE']) 
    for ticker in My_holdings:
        status,predictions,o_last,rmse = stock_status(Ticker = ticker,raw_data=raw_data_all.copy(),\
                 last_price=last_price_all.copy(),epochs=40000, batch_size=512,verbose = 0)
        time.sleep(0.1)
        STATUS[ticker] = (status,predictions,o_last)
        csvwriter.writerow([ticker,status,np.round(predictions,2),np.round(o_last,2),np.round(rmse,2)])