#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from matplotlib import pyplot as plt


# In[4]:


from sklearn import model_selection


# In[5]:


from sklearn.metrics import confusion_matrix


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


from sklearn.preprocessing import MinMaxScaler


# In[9]:


from keras.models import Sequential


# In[10]:


import keras 


# In[11]:


get_ipython().system('pip install keras')


# In[12]:


from keras.models import Sequential


# In[13]:


get_ipython().system('pip install tensorflow')


# In[14]:


from keras.models import Sequential


# In[15]:


from keras.layers import Dense


# In[16]:


from keras.layers import LSTM


# In[17]:


from keras.layers import Dropout


# In[18]:


df = pd.read_csv("NFLX.csv")


# In[19]:


df.head(10)


# In[20]:


#training and testing the data


# In[21]:


df.shape


# In[22]:


df1_train = df.reset_index()['Close']


# In[23]:


df1_train.head()


# In[24]:


df1_train.shape


# In[25]:


plt.plot(df1_train)


# In[26]:


#train = df.iloc[:800,1:2].values


# In[27]:


#test = df.iloc[800:,1:2].values


# In[28]:


#scaling the data so as to make it fit.


# In[29]:


sc = MinMaxScaler(feature_range = (0,1))


# In[30]:


df1_train = sc.fit_transform(np.array(df1_train).reshape(-1,1))


# In[31]:


df1_train


# In[32]:


#train-test split using cross validation 


# In[33]:


train_size = int(len(df1_train)*0.65)
test_size = len(df1_train)-train_size
train_data=df1_train[0:train_size,:]
test_data = df1_train[train_size:len(df1_train),:1]


# In[34]:


train_size, test_size


# In[35]:


#data preprocessing


# In[36]:


import numpy
def create_ds(dataset, time_step=1):
    x,y = [], []
    for i in range(len(dataset)-time_step-1):
        #i = 0...100
        a = dataset[i:(i+time_step), 0]
        x.append(a)
        y.append(dataset[i+time_step,0])
    return numpy.array(x), numpy.array(y)


# In[37]:


time_step = 100
x_train, y_train = create_ds(train_data, time_step)
x_test, y_test = create_ds(test_data, time_step)


# In[38]:


print(x_train.shape), print(y_train.shape)


# In[39]:


print(x_test.shape), print(y_test.shape)


# In[40]:


#3d
#reshape input to be (samples, time_steps, features) which is rquired for LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)


# In[41]:


#lstm model building


# In[42]:


algo = Sequential()
algo.add(LSTM(units = 50, return_sequences = True, input_shape = (100,1)))
algo.add(LSTM(units = 50, return_sequences = True))
algo.add(LSTM(units = 50))
algo.add(Dense(units = 1))
algo.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[43]:


algo.summary()


# In[44]:


algo.fit(x_train, y_train, validation_data = (x_test,y_test),epochs = 50, batch_size =64, verbose = 1)


# In[45]:


#prediction


# In[46]:


train_pred=algo.predict(x_train)
test_pred=algo.predict(x_test)


# In[47]:


#transform back to original form


# In[48]:


train_pred = sc.inverse_transform(train_pred)
test_pred = sc.inverse_transform(test_pred)


# In[49]:


#calculate RMSE perfomance metric


# In[50]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_pred))


# In[51]:


#test data RMSE
math.sqrt(mean_squared_error(y_test,test_pred))


# In[52]:


look_back = 100
trainpredictplot = numpy.empty_like(df1_train)
trainpredictplot[:, :] = np.nan
trainpredictplot[look_back:len(train_pred)+ look_back, :] = train_pred
#shift test prediction for plotting
testpredictplot = numpy.empty_like(df1_train)
testpredictplot[:, :] = np.nan
testpredictplot[len(train_pred)+ (look_back*2)+1:len(df1_train)-1, :] = test_pred
#plot baseline and prediction
plt.plot(sc.inverse_transform(df1_train))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.show()


# In[53]:


len(test_data)


# In[55]:


x_input = test_data[1666:].reshape(1,-1)
x_input.shape


# In[56]:


temp_input = list(x_input)
temp_input = temp_input[0].tolist()


# In[57]:


from numpy import array

list_output = []
n_steps = 100
i=0
while(i<30):
    if(len(temp_input)>100):
        #print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = algo.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        #print(temp_input)
        list_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = algo.predict(x_input, verbose =0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        list_output.extend(yhat.tolist())
        i=i+1
        
print(list_output)


# In[58]:


day_new = np.arange(1,101)
day_pred = np.arange(101,131)


# In[59]:


import matplotlib.pyplot as plt


# In[60]:


len(df1_train)-100


# In[61]:


plt.plot(day_new, sc.inverse_transform(df1_train[4944:]))
plt.plot(day_pred, sc.inverse_transform(list_output))


# In[62]:


df3 = df1_train.tolist()
df3.extend(list_output)
plt.plot(df3[4000:])


# In[63]:


df3 = sc.inverse_transform(df3).tolist()
plt.plot(df3)


# ## This way we can predict the future stock prices
