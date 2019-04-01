
# coding: utf-8

# In[1]:


#import the needed module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# -- coding: utf-8 --


# ### Loading needed data and concating them

# In[3]:


#loading data function
def load(data_input):
    data=pd.read_csv(data_input,encoding = 'gb18030')
    data['Date'] = pd.to_datetime(data.Date , format = '%Y/%m/%d')
    df = data.drop(['Date'], axis=1)
    df.index = data.Date
    #creat more time feature
    df['date']=pd.to_datetime(df['date'], format='%Y%m%d')
    df['month']=df['date'].dt.month
    df['day']=df['date'].dt.dayofweek+1
    df['date']=df['date'].dt.day
    return df


# In[4]:


#loading electricity data
df17_18=load('2017-2018electricity.csv')
df18_19=load('2018-2019electricity.csv')


# In[5]:


#show the data
df17_18


# In[6]:


#we are insterested in 'date' 'month' 'day' 'Peak Load(MW)' 
#check if there are any N/A in 'date' 'month' 'day' 'Peak Load(MW)' 
print(df17_18[['date','month','day','Peak Load(MW)']].isnull().any())
print(df18_19[['date','month','day','Peak Load(MW)']].isnull().any())


# In[7]:


#load the holliday data
hd17=pd.read_excel('2017holliday.xls')
hd18_19=pd.read_excel('2018-2019holliday.xlsx')
#hd17['holliday']


# In[8]:


#take 'Date' as index, 'holliday' as column
hd17['Date']=pd.to_datetime(hd17['Date'], format='%Y%m%d')
hd_17=pd.DataFrame(data=np.array(hd17['holliday']),index=hd17['Date'],columns=['holliday'])
hd18_19['Date']=pd.to_datetime(hd18_19['Date'], format='%Y%m%d')
hd_1819=pd.DataFrame(data=np.array(hd18_19['holliday']),index=hd18_19['Date'],columns=['holliday'])


# In[9]:


#concad the electricity data and holliday data
df17=df17_18.loc["2017-01-01":"2017-12-31"]
df17=pd.concat([df17,hd_17],axis=1)
df18_19=pd.concat([df18_19,hd_1819],axis=1)
df=pd.concat([df17,df18_19],axis=0)
df


# In[10]:


#check if there are any N/A in 'Date' 'date' 'month' 'day' 'Peak Load(MW)' 'holliday'
print(df[['date','month','day','Peak Load(MW)','holliday']].isnull().any())


# In[11]:


#see the distribution of "peak Load" v.s "Date"
plt.figure(figsize=([7.5, 4.8]))
plt.plot(df['Peak Load(MW)'])
plt.vlines(x=['2017-12-31','2018-12-31'],ymin=20000,ymax=37500,color='r')


# ### Strationary time series testing
# reference：http://xtf615.com/2017/03/08/Python%E5%AE%9E%E7%8E%B0%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E5%88%86%E6%9E%90/

# In[12]:


#穩定性檢測：滾動統計
def rolling_statistics(timeseries):
    #Determing rolling statistics
    rolmean=timeseries.rolling(50).mean()
    rolstd=timeseries.rolling(50).std()

    #Plot rolling statistics:
    plt.figure(figsize=([7.5, 4.8]))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='upper right')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
#穩定性檢測：ADF檢驗
#H0: this time series isn't stable.
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    rolling_statistics(timeseries)
    print('Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

df_pl=df['Peak Load(MW)']
adf_test(df_pl)


# from above, both p-value and Test Statistic > Critical Value 
# 
# => this time series isn't stable
# 
# => usually related to trend and seasonal of time series

# In[13]:


#disassemble data into trend, seasonal, residule
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition_origin = seasonal_decompose(df_pl)
trend_origin = decomposition_origin.trend
seasonal_origin = decomposition_origin.seasonal
residual_origin = decomposition_origin.resid

plt.subplot(411)
plt.plot(df_pl,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend_origin, label='Trend')
plt.legend(loc='best')
plt.subplot(413);
plt.plot(seasonal_origin,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual_origin, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[14]:


#check if the trend exist (H0:unstable)
df_pl_decompose_t=trend_origin
df_pl_decompose_t.dropna(inplace=True)
adf_test(df_pl_decompose_t)
#from result, we can see that both p-value and Test Statistic > Critical Value => trend exist


# In[15]:


#check if the seasonal exist (H0:unstable)
df_pl_decompose_s=seasonal_origin
df_pl_decompose_s.dropna(inplace=True)
adf_test(df_pl_decompose_s)
#from result, we can see that both p-value and Test Statistic > Critical Value => seasonal exist


# In[17]:


#stationary processing: difference(差分)) => eliminate the trend and the seasonal
#log to punish the larger value
df_pl_log=np.log(df_pl)
#difference step = 1
df_pl_log_diff = df_pl_log - df_pl_log.shift(periods=1)
plt.figure(figsize=([7.5, 4.8]))
plt.plot(df_pl_log_diff)


# In[18]:


#test the stastionary of the step-1-difference data
df_pl_log_diff=df_pl_log_diff[df_pl_log_diff.isnull()==False]
adf_test(df_pl_log_diff)


# from above, Test Statistic < Critical Value => stable

# ### because the stationary, let's biild the model: LSTM and ARIMA

# #### LSTM model

# In[19]:


df_lstm=df[['date','Peak Load(MW)','month','day','holliday']]
df_lstm.head()


# In[20]:


#data processing for fitting the lstm model

#dummy variables
df_dum=pd.get_dummies(df_lstm,columns=['date','month','day','holliday'],drop_first=True)

#normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_n_d=scaler.fit_transform(df_dum)


# In[21]:


df_n_d.shape


# In[22]:


#building the X,Y training/testing data
#many to many (pastDay=futureDay)
def buildX_Y(df, pastDay, futureDay):
    X_train, Y_train = [], []
    for i in range(df.shape[0]-futureDay-pastDay):
        X_train.append(np.array(df[i:i+pastDay]))
        Y_train.append(np.array(df[i+pastDay:i+pastDay+futureDay,0]))
    return np.array(X_train), np.array(Y_train)
pastDay=8
futureDay=8
X_train,Y_train=buildX_Y(df_n_d,pastDay,futureDay)
Y_train = Y_train[:,:,np.newaxis] #shape needs to be [samples, time steps, features]
print(X_train.shape,Y_train.shape)


# In[23]:


#split the data to training data and testing data
train_size = int(len(X_train) * 0.67)
test_size = len(X_train) - train_size
X_train, X_test = X_train[0:train_size], X_train[train_size:len(X_train)]
Y_train,Y_test=Y_train[0:train_size], Y_train[train_size:len(Y_train)]

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)


# In[24]:


#build the LSTM model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
model = Sequential()
model.add(LSTM(50, input_length=X_train.shape[1], input_dim=X_train.shape[2],return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
history=model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_test,Y_test),verbose=2,shuffle=False)


# In[25]:


#see the performance of prediction
#model prediction
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)


# In[26]:


#inverse transform the data

#MinMaxScaler fit just 'Peak load(MW)' to inverse Y-data 
df__pl=df_dum['Peak Load(MW)']
df__pl=df_pl[:,np.newaxis]
scaler = MinMaxScaler(feature_range=(0, 1))
df_n_d_0=scaler.fit_transform(df__pl)

trainPredict=np.squeeze(trainPredict, axis=2)
trainPredict = scaler.inverse_transform(trainPredict)
Y_train=np.squeeze(Y_train, axis=2)
Y_train = scaler.inverse_transform(Y_train)
testPredict=np.squeeze(testPredict, axis=2)
testPredict = scaler.inverse_transform(testPredict)
Y_test=np.squeeze(Y_test, axis=2)
Y_test = scaler.inverse_transform(Y_test)


# In[27]:


#see the performance of prediction
#RMSE score
import math
from sklearn.metrics import mean_squared_error
trainScore = math.sqrt(mean_squared_error(Y_train[:,0], trainPredict[:,0]))
print('Train Score: %.4f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_test[:,0], testPredict[:,0]))
print('Test Score: %.4f RMSE' % (testScore))


# #### ARIMA model

# In[28]:


#ARIMA need proper p、q
#這個cell會跑很久，所以我command起來，將其output出的p、q值直接放到下一個cell用
"""
import sys
from statsmodels.tsa.arima_model import ARMA
def _proper_model(df_log_diff, maxLag):
    best_p = 0 
    best_q = 0
    best_bic = sys.maxsize
    best_model=None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(df_log_diff, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARMA.bic
            print(bic, best_bic)
            if bic < best_bic:
                best_p = p
                best_q = q
                best_bic = bic
                best_model = results_ARMA
    return best_p,best_q,best_model
_proper_model(df_pl_log_diff, 10)
"""


# In[31]:


from statsmodels.tsa.arima_model import ARIMA
df_ARIMA=df_pl_log[:'2019-03-20']
model = ARIMA(df_ARIMA, order=(6, 1, 7))  #arg:(p, diff-step, q)
results_ARIMA = model.fit(disp=-1) 
plt.plot(df_pl_log_diff[:'2019-03-20'])
plt.plot(results_ARIMA.fittedvalues, color='red')#fittedvalues is the result of difference, so it's needed to recover data back 
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-df_pl_log_diff[:'2019-03-20'])**2))


# In[32]:


#recover the data

#recover the fittedvalues value from difference to log value
def predict_diff_recover(log,predict_value, d):
    shift_log_list = []
    last_data_shift_list = []
    tmp_log = log
    for i in d:
        last_data_shift_list.append(tmp_log[-i])
        print(last_data_shift_list)
        shift_log = tmp_log.shift(i)
        shift_log_list.append(shift_log)
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_log_list[-i-1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data 
diff_recover_log = predict_diff_recover(df_ARIMA,results_ARIMA.fittedvalues, d=[1])

#recover the log data to origin data
log_recover = np.exp(diff_recover_log)


# In[33]:


from sklearn.metrics import mean_squared_error
import math
math.sqrt(mean_squared_error(log_recover,df_pl['2017-01-02':'2019-03-20']))
plt.figure(figsize=([7.5, 4.8]))
plt.plot(df_pl['2017-01-02':'2019-03-20'])
plt.plot(log_recover)
plt.title('RMSE:%.4f' %math.sqrt(mean_squared_error(log_recover,df_pl['2017-01-02':'2019-03-20'])))


# In[34]:


#testing: prediction performance
forecast_n =11  
forecast_ARIMA_log = results_ARIMA.forecast(forecast_n)
forecast_ARIMA_log = forecast_ARIMA_log[0]

forecast_date=pd.date_range('2019-03-21',periods=11)
prediction_forecast_ARIMA_log=pd.Series(forecast_ARIMA_log,index=forecast_date)
prediction_forecast_ARIMA=np.exp(prediction_forecast_ARIMA_log)
print(prediction_forecast_ARIMA)


# In[35]:


print('RMSE:%.4f' %math.sqrt(mean_squared_error(prediction_forecast_ARIMA,df['Peak Load(MW)']['2019-03-21':'2019-03-31'])))


# I find that ARIMA model performs better than LSTM model.
# 
# =>predict the electricity of 2019-04-02 ~ 2019-04-08 by ARIMA

# In[36]:


#build and fit the ARIMA model
model = ARIMA(df_pl_log, order=(6, 1, 7))  #arg:(p, diff-step, q)
results_ARIMA = model.fit(disp=-1) 
#recover the data
diff_recover_log = predict_diff_recover(df_pl_log,results_ARIMA.fittedvalues, d=[1])
log_recover = np.exp(diff_recover_log)


# In[37]:


#prediction
forecast_n =8  
forecast_ARIMA_log = results_ARIMA.forecast(forecast_n)
forecast_ARIMA_log = forecast_ARIMA_log[0]
forecast_ARIMA=np.exp(forecast_ARIMA_log).astype(int)
forecast_date=range(20190402,20190409)
prediction_forecast_ARIMA=pd.DataFrame({'peak_load(MW)':forecast_ARIMA[1:]},index=forecast_date)
prediction_forecast_ARIMA.index.name='date'

print(prediction_forecast_ARIMA)


# In[38]:


prediction_forecast_ARIMA.to_csv('submission.csv')

