#!/usr/bin/env python
# coding: utf-8

# In[78]:


import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[79]:


data1 = pd.read_csv('CGMDatenumLunchPat1.csv')
print('Number of data points data1: ', data1.shape[0])
print('Number of features data1: ', data1.shape[1])
data2 = pd.read_csv('CGMDatenumLunchPat2.csv')
print('Number of data points data2: ', data2.shape[0])
print('Number of features data2: ', data2.shape[1])
data3 = pd.read_csv('CGMDatenumLunchPat3.csv')
print('Number of data points data3: ', data3.shape[0])
print('Number of features data3: ', data3.shape[1])
data4 = pd.read_csv('CGMDatenumLunchPat4.csv')
print('Number of data points data4: ', data4.shape[0])
print('Number of features data4: ', data4.shape[1])
data5 = pd.read_csv('CGMDatenumLunchPat5.csv')
print('Number of data points data5: ', data5.shape[0])
print('Number of features data5: ', data5.shape[1])


# In[80]:


data1.head()


# In[81]:


data1.tail()


# In[82]:


data6 = pd.read_csv('CGMSeriesLunchPat1.csv')
print(data6.head())


# In[83]:


data6 = pd.read_csv('CGMSeriesLunchPat1.csv')
data6 = data6.iloc[:, :-1]
print(data6.shape)
print('Number of data points data1: ', data6.shape[0])
print('Number of features data1: ', data6.shape[1])
data7 = pd.read_csv('CGMSeriesLunchPat2.csv')
data7 = data7.iloc[:, :-1]
print('Number of data points data2: ', data7.shape[0])
print('Number of features data2: ', data7.shape[1])
data8 = pd.read_csv('CGMSeriesLunchPat3.csv')
data8 = data8.iloc[:, :-1]
print('Number of data points data3: ', data8.shape[0])
print('Number of features data3: ', data8.shape[1])

data10 = pd.read_csv('CGMSeriesLunchPat5.csv')
data10 = data10.iloc[:, :-1]
print('Number of data points data5: ', data10.shape[0])
print('Number of features data5: ', data10.shape[1])


# In[84]:


data9.columns.values


# In[85]:


data9 = pd.read_csv('CGMSeriesLunchPat4.csv')
data9.drop(data9.iloc[:, 30:42], inplace = True, axis = 1) 

print('Number of data points data: ', data9.shape[0])
print('Number of features data: ', data9.shape[1])


# In[86]:


data6.fillna(data6.mean(), inplace = True)
data7.fillna(data7.mean(), inplace = True)
data8.fillna(data8.mean(), inplace = True)
data9.fillna(data9.mean(), inplace = True)
data10.fillna(data10.mean(), inplace = True)


# # Statistical Feature Extraction

# # Mean

# ### data6 (cgmseries 1)

# In[87]:


K = [X for X in range(0,150,5)]
mean1 = data6.mean(axis=0)
plt.figure(figsize=(5,4))
plt.style.use('default')
plt.plot(K,mean1)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Mean for CGM_series_6",fontsize = 12)


# In[88]:


mod_data6 = data6.transpose()
mod_data6["mean"] = mean1
print(mod_data6.columns.values)


# ### Data 7 (cgmseries 2)

# In[89]:


mean2 = data7.mean(axis=0)
plt.figure(figsize=(5,4))
plt.style.use('default')
plt.plot(K,mean2)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Mean for CGM_series_7",fontsize = 12)


# In[90]:


mod_data7 = data7.transpose()
mod_data7["mean"] = mean2
print(mod_data7.columns.values)


# ### Data8 (cgmseries 3)

# In[91]:


mean3 = data8.mean(axis=0)
plt.figure(figsize=(5,4))
plt.style.use('default')
plt.plot(K,mean3)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Mean for CGM_series_8",fontsize = 12)


# In[92]:


mod_data8 = data8.transpose()
mod_data8["mean"] = mean3
print(mod_data8.columns.values)


# ### Data9(cgmseries 4)

# In[93]:


mean4 = data9.mean(axis=0)
plt.figure(figsize=(5,4))
plt.style.use('default')
plt.plot(K,mean4)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Mean for CGM_series_9",fontsize = 12)


# In[94]:


mod_data9 = data9.transpose()
mod_data9["mean"] = mean4
print(mod_data9.columns.values)


# ### Data10 (cgmseries 5)

# In[95]:


mean5 = data10.mean(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,mean5)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Mean for CGM_series_10",fontsize = 12)


# In[96]:


mod_data10 = data10.transpose()
mod_data10["mean"] = mean3
print(mod_data10.columns.values)


# # Median

# ### Data6(cgmseries 1)

# In[97]:


median1 = data6.median(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,median1)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Median for CGM_series_6",fontsize = 12)


# In[98]:


mod_data6["median"] = median1
print(mod_data6.columns.values)


# ### Data7(cgmseries 2)

# In[99]:


median2 = data7.median(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,median2)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Median for CGM_series_7",fontsize = 12)


# In[100]:


mod_data7["median"] = median2
print(mod_data7.columns.values)


# ### Data9 (cgmseries 4)

# In[101]:


median3 = data8.median(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,median3)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Medain for CGM_series_8",fontsize = 12)


# In[102]:


mod_data8["median"] = median3
print(mod_data8.columns.values)


# ### Data9 (cgmseries 4)

# In[103]:


median4 = data9.median(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,median4)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Mean for CGM_series_9",fontsize = 12)


# In[104]:


mod_data9["median"] = median4
print(mod_data9.columns.values)


# ### Data10 (cgmseries 5)

# In[105]:


median5 = data10.median(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,median5)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Median for CGM_series_10",fontsize = 12)


# In[106]:


mod_data10["median"] = median5
print(mod_data9.columns.values)


# # Variance

# ### data6 (cgmseries 1)

# In[107]:


variance1 = data6.var(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,variance1)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("variance for CGM_series_6",fontsize = 12)


# In[108]:


mod_data6["variance"] = variance1
print(mod_data6.columns.values)


# ### data7 (cgmseries 2) 

# In[109]:


variance2 = data7.var(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,variance2)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("variance for CGM_series_7",fontsize = 12)


# In[110]:


mod_data7["variance"] = variance2
print(mod_data7.columns.values)


# ### data8 (cgmseries 3) 

# In[111]:


variance3 = data8.var(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,variance3)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("variance for CGM_series_8",fontsize = 12)


# In[112]:


mod_data8["variance"] = variance3
print(mod_data8.columns.values)


# ### data 9 (cgmseries 4)

# In[113]:


variance4 = data9.var(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,variance4)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("variance for CGM_series_9",fontsize = 12)


# In[114]:


mod_data9["variance"] = variance4
print(mod_data9.columns.values)


# ### Data10 (cgmseries 5)

# In[115]:


variance5 = data10.var(axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,variance5)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("variance for CGM_series_10",fontsize = 12)


# In[116]:


mod_data10["variance"] = variance5
print(mod_data10.columns.values)


# # Moving Averages

# # Simple Moving average

# ### Data6 (cgmseries 1)

# In[117]:


mSMA_data6 = data6.iloc[[1],:]
SMA_data6 = mSMA_data6.transpose()
K = [X for X in range(0,150,5)]
SMA6 = SMA_data6.iloc[:,:].rolling(window=3).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,SMA6 )
plt.xlabel("Time",fontsize = 12)
plt.ylabel("SMA for CGM_series_6",fontsize = 12)


# In[118]:


mod_data6["SMA"] = SMA6
print(mod_data6.columns.values)
print(mod_data6.shape)


# ### Data7 (cgmseries 2)

# In[119]:


mSMA_data7 = data7.iloc[[1],:]
SMA_data7 = mSMA_data7.transpose()
SMA7 = SMA_data7.iloc[:,:].rolling(window=3).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,SMA7 )
plt.xlabel("Time",fontsize = 12)
plt.ylabel("SMA for CGM_series_7",fontsize = 12)


# In[120]:


mod_data7["SMA"] = SMA7
print(mod_data7.columns.values)


# ### data 8 (cgmseries 3)

# In[230]:


mSMA_data8 = data8.iloc[[1],:]
SMA_data8 = mSMA_data8.transpose()
SMA8 = SMA_data8.iloc[:,:].rolling(window=3).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,SMA8)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("SMA for CGM_series_8",fontsize = 12)


# In[122]:


mod_data8["SMA"] = SMA8
print(mod_data8.columns.values)


# ### Data 9 (cgmseries 4)

# In[231]:


mSMA_data9 = data9.iloc[[1],:]
SMA_data9 = mSMA_data9.transpose()
SMA9 = SMA_data9.iloc[:,:].rolling(window=3).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,SMA9)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("SMA for CGM_series_9",fontsize = 12)


# In[124]:


mod_data9["SMA"] = SMA9
print(mod_data9.columns.values)


# ### Data 10 (cgmseries 5)

# In[125]:


mSMA_data10 = data10.iloc[[1],:]
SMA_data10 = mSMA_data10.transpose()
SMA10 = SMA_data10.iloc[:,:].rolling(window=3).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,SMA10)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("SMA for CGM_series_7",fontsize = 12)


# In[126]:


mod_data10["SMA"] = SMA10
print(mod_data10.columns.values)


# # Exponential Moving Average

# ### Data6 (cgmseries 1)

# In[127]:


mEMA_data6 = data6.iloc[[1],:]
EMA_data6 = mEMA_data6.transpose()
EMA6 = EMA_data6.iloc[:,:].ewm(span=40,adjust=False).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,EMA6)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("EMA for CGM_series_6",fontsize = 12)


# In[128]:


mod_data6["EMA"] = EMA6
print(mod_data6.columns.values)


# ### Data 7 (cgmseries 2)

# In[129]:


mEMA_data7 = data7.iloc[[1],:]
EMA_data7 = mEMA_data7.transpose()
EMA7 = EMA_data7.iloc[:,:].ewm(span=40,adjust=False).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,EMA7)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("EMA for CGM_series_7",fontsize = 12)


# In[130]:


mod_data7["EMA"] = EMA6
print(mod_data7.columns.values)


# ### Data 8 (cgmseries 3)

# In[131]:


mEMA_data8 = data8.iloc[[1],:]
EMA_data8 = mEMA_data8.transpose()
EMA8 = EMA_data8.iloc[:,:].ewm(span=40,adjust=False).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,EMA8)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("EMA for CGM_series_8",fontsize = 12)


# In[132]:


mod_data8["EMA"] = EMA8
print(mod_data8.columns.values)


# ### Data 9 (cgmseries 4)

# In[133]:


mEMA_data9 = data9.iloc[[1],:]
EMA_data9 = mEMA_data9.transpose()
print(EMA_data9.shape)
EMA10 = EMA_data9.iloc[:,:].ewm(span=40,adjust=False).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,EMA10)
plt.xlabel("Time",fontsize = 12)
plt.ylabel("EMA for CGM_series_9",fontsize = 12)


# In[134]:


mod_data10["EMA"] = EMA10
print(mod_data10.columns.values)


# ### Data 10 (cgmseries 5)

# In[135]:


mEMA_data10 = data10.iloc[[1],:]
EMA_data10 = mEMA_data10.transpose()
print(EMA_data10.shape)
EMA_data10['pandas_SMA_6'] = EMA_data10.iloc[:,:].ewm(span=40,adjust=False).mean()
plt.style.use('default')
plt.figure(figsize=(5,4))
plt.plot(K,EMA_data10['pandas_SMA_6'])
plt.xlabel("Time",fontsize = 12)
plt.ylabel("EMA for CGM_series_10",fontsize = 12)


# In[136]:


mod_data10["EMA"] = EMA10
print(mod_data10.columns.values)


# # FFT

# ### Data 6 (cgmseries 1)

# In[225]:


# import numpy as np
import pandas as pd

K =[ X for X in range(0,150,5)]
df1 = data6.apply(np.fft.fft, axis=0)
print(df1.shape)
plt.style.use('default')
plt.figure(figsize=(5,4))
new = df1.iloc[[1],:]
new1 = new.stack().tolist()
print(len(new1))
print(new1)
plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("FFT",fontsize = 12)
plt.plot(K,new1)


# In[160]:


mod_data6["FFT"] = new1
print(mod_data6.columns.values)


# ### Data 7 (cgm series 2)

# In[226]:


import numpy as np
import pandas as pd

K =[ X for X in range(0,150,5)]
df2 = data7.apply(np.fft.fft, axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
new = df2.iloc[[1],:]
new2 = new.stack().tolist()
print(len(new2))
print(new2)
plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("FFT",fontsize = 12)
plt.plot(K,new2)


# In[162]:


mod_data7["FFT"] = new2
print(mod_data7.columns.values)


# ### Data 8 (cgmseries 3)

# In[227]:


import numpy as np
import pandas as pd

K =[ X for X in range(0,150,5)]
df3 = data8.apply(np.fft.fft, axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
new = df3.iloc[[1],:]
new3 = new.stack().tolist()
print(len(new3))
print(new3)
plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("FFT",fontsize = 12)
plt.plot(K,new3)


# In[164]:


mod_data8["FFT"] = new3
print(mod_data8.columns.values)


# ### Data 9 (cgmseries 4) 

# In[228]:


import numpy as np
import pandas as pd

K =[ X for X in range(0,150,5)]
df4 = data9.apply(np.fft.fft, axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
new = df4.iloc[[1],:]
new4 = new.stack().tolist()
print(len(new4))
print(new4)
plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("FFT",fontsize = 12)
plt.plot(K,new4)


# In[167]:


mod_data9["FFT"] = new4
print(mod_data9.columns.values)


# ### Data 10 (cgmseries 5)

# In[229]:


import numpy as np
import pandas as pd

K =[ X for X in range(0,150,5)]
df5 = data10.apply(np.fft.fft, axis=0)
plt.style.use('default')
plt.figure(figsize=(5,4))
new = df5.iloc[[1],:]
new5 = new.stack().tolist()
print(len(new5))
print(new5)
plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("FFT",fontsize = 12)
plt.plot(K,new5)


# In[169]:


mod_data10["FFT"] = new5
print(mod_data10.columns.values)


# # Polynomial Fitting

# # Degree 9

# ### Data6 (cgmseries1)

# In[214]:


coeff_data6_1 = data6.iloc[[1],:]
new1 = coeff_data6_1.stack().tolist()
coeffs_data6_1 = np.polyfit(K,new1,9)
print(coeffs_data6_1)

coeff_data6_2 = data6.iloc[[2],:]
new2 = coeff_data6_2.stack().tolist()
coeffs_data6_2 = np.polyfit(K,new2,9)
print(coeffs_data6_2)

coeff_data6_3 = data6.iloc[[3],:]
new3 = coeff_data6_3.stack().tolist()
coeffs_data6_3 = np.polyfit(K,new3,9)
print(coeffs_data6_3)

coeff_data6_4 = data6.iloc[[4],:]
new4 = coeff_data6_4.stack().tolist()
coeffs_data6_4 = np.polyfit(K,new4,9)
print(coeffs_data6_4)

coeff_data6_5 = data6.iloc[[5],:]
new5 = coeff_data6_5.stack().tolist()
coeffs_data6_5 = np.polyfit(K,new5,9)
print(coeffs_data6_5)

coeff_data6_6 = data6.iloc[[6],:]
new6 = coeff_data6_6.stack().tolist()
coeffs_data6_6 = np.polyfit(K,new6,9)
print(coeffs_data6_6)

coeff_data6_7 = data6.iloc[[7],:]
new7 = coeff_data6_7.stack().tolist()
coeffs_data6_7 = np.polyfit(K,new7,9)
print(coeffs_data6_7)

coeff_data6_8 = data6.iloc[[8],:]
new8 = coeff_data6_8.stack().tolist()
coeffs_data6_8 = np.polyfit(K,new8,9)
print(coeffs_data6_8)

coeff_data6_9 = data6.iloc[[9],:]
new9 = coeff_data6_9.stack().tolist()
coeffs_data6_9 = np.polyfit(K,new9,9)
print(coeffs_data6_9)

coeff_data6_10 = data6.iloc[[10],:]
new10 = coeff_data6_10.stack().tolist()
coeffs_data6_10 = np.polyfit(K,new10,9)
print(coeffs_data6_10)


# In[215]:


coeff_data6_11 = data6.iloc[[11],:]
new11 = coeff_data6_11.stack().tolist()
coeffs_data6_11 = np.polyfit(K,new11,9)
print(coeffs_data6_11)

coeff_data6_12 = data6.iloc[[12],:]
new12 = coeff_data6_12.stack().tolist()
coeffs_data6_12 = np.polyfit(K,new12,9)
print(coeffs_data6_12)

coeff_data6_13 = data6.iloc[[13],:]
new13 = coeff_data6_13.stack().tolist()
coeffs_data6_13 = np.polyfit(K,new13,9)
print(coeffs_data6_13)

coeff_data6_14 = data6.iloc[[14],:]
new14 = coeff_data6_14.stack().tolist()
coeffs_data6_14 = np.polyfit(K,new14,9)
print(coeffs_data6_14)

coeff_data6_15 = data6.iloc[[5],:]
new15 = coeff_data6_15.stack().tolist()
coeffs_data6_15 = np.polyfit(K,new15,9)
print(coeffs_data6_15)

coeff_data6_16 = data6.iloc[[16],:]
new16 = coeff_data6_16.stack().tolist()
coeffs_data6_16 = np.polyfit(K,new16,9)
print(coeffs_data6_16)

coeff_data6_17 = data6.iloc[[17],:]
new17 = coeff_data6_17.stack().tolist()
coeffs_data6_17 = np.polyfit(K,new17,9)
print(coeffs_data6_17)

coeff_data6_18 = data6.iloc[[18],:]
new18 = coeff_data6_18.stack().tolist()
coeffs_data6_18 = np.polyfit(K,new18,9)
print(coeffs_data6_18)

coeff_data6_19 = data6.iloc[[19],:]
new19 = coeff_data6_19.stack().tolist()
coeffs_data6_19 = np.polyfit(K,new19,9)
print(coeffs_data6_19)

coeff_data6_20 = data6.iloc[[20],:]
new20 = coeff_data6_20.stack().tolist()
coeffs_data6_20 = np.polyfit(K,new20,9)
print(coeffs_data6_20)


# In[216]:


coeff_data6_21 = data6.iloc[[21],:]
new21 = coeff_data6_21.stack().tolist()
coeffs_data6_21 = np.polyfit(K,new21,9)
print(coeffs_data6_21)

coeff_data6_22 = data6.iloc[[22],:]
new22 = coeff_data6_22.stack().tolist()
coeffs_data6_22 = np.polyfit(K,new22,9)
print(coeffs_data6_22)

coeff_data6_23 = data6.iloc[[23],:]
new23 = coeff_data6_23.stack().tolist()
coeffs_data6_23 = np.polyfit(K,new23,9)
print(coeffs_data6_23)

coeff_data6_24 = data6.iloc[[24],:]
new24 = coeff_data6_24.stack().tolist()
coeffs_data6_24 = np.polyfit(K,new24,9)
print(coeffs_data6_24)

coeff_data6_25 = data6.iloc[[25],:]
new25 = coeff_data6_25.stack().tolist()
coeffs_data6_25 = np.polyfit(K,new25,9)
print(coeffs_data6_25)

coeff_data6_26 = data6.iloc[[26],:]
new26 = coeff_data6_26.stack().tolist()
coeffs_data6_26 = np.polyfit(K,new26,9)
print(coeffs_data6_26)

coeff_data6_27 = data6.iloc[[27],:]
new27 = coeff_data6_27.stack().tolist()
coeffs_data6_27 = np.polyfit(K,new27,9)
print(coeffs_data6_27)

coeff_data6_28 = data6.iloc[[28],:]
new28 = coeff_data6_28.stack().tolist()
coeffs_data6_28 = np.polyfit(K,new28,9)
print(coeffs_data6_28)

coeff_data6_29 = data6.iloc[[29],:]
new29 = coeff_data6_29.stack().tolist()
coeffs_data6_29 = np.polyfit(K,new29,9)
print(coeffs_data6_29)

coeff_data6_30 = data6.iloc[[30],:]
new30 = coeff_data6_30.stack().tolist()
coeffs_data6_30 = np.polyfit(K,new30,9)
print(coeffs_data6_30)


# In[217]:



poly_mod_1 = pd.DataFrame()
poly_mod_1["coef1"] = coeffs_data6_1
poly_mod_1["coef2"] = coeffs_data6_2
poly_mod_1["coef3"] = coeffs_data6_3
poly_mod_1["coef4"] = coeffs_data6_4
poly_mod_1["coef5"] = coeffs_data6_5
poly_mod_1["coef6"] = coeffs_data6_6
poly_mod_1["coef7"] = coeffs_data6_7
poly_mod_1["coef8"] = coeffs_data6_8
poly_mod_1["coef9"] = coeffs_data6_9
poly_mod_1["coef10"] = coeffs_data6_10
poly_mod_1["coef11"] = coeffs_data6_11
poly_mod_1["coef12"] = coeffs_data6_12
poly_mod_1["coef13"] = coeffs_data6_13
poly_mod_1["coef14"] = coeffs_data6_14
poly_mod_1["coef15"] = coeffs_data6_15
poly_mod_1["coef16"] = coeffs_data6_16
poly_mod_1["coef17"] = coeffs_data6_17
poly_mod_1["coef18"] = coeffs_data6_18
poly_mod_1["coef19"] = coeffs_data6_19
poly_mod_1["coef20"] = coeffs_data6_20
poly_mod_1["coef21"] = coeffs_data6_21
poly_mod_1["coef22"] = coeffs_data6_22
poly_mod_1["coef23"] = coeffs_data6_23
poly_mod_1["coef24"] = coeffs_data6_24
poly_mod_1["coef25"] = coeffs_data6_25
poly_mod_1["coef26"] = coeffs_data6_26
poly_mod_1["coef27"] = coeffs_data6_27
poly_mod_1["coef28"] = coeffs_data6_28
poly_mod_1["coef29"] = coeffs_data6_29
poly_mod_1["coef30"] = coeffs_data6_30
print(poly_mod_1.shape)


# In[218]:


p = np.poly1d(coeffs_data6_1)


# In[220]:


plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Polynomial Fitting",fontsize = 12)
plt.plot(K,p(K))


# # Data 7

# In[221]:


mcoeff_data7 = data7.iloc[[1],:]
print(mcoeff_data7.shape)
new = mcoeff_data7.stack().tolist()
coeffs_data7 = np.polyfit(K,new,9)

print(coeffs_data7)
p = np.poly1d(coeffs_data7)
plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Polynomial Fitting",fontsize = 12)
plt.plot(K,p(K))


# # Data 8

# In[222]:


mcoeff_data8 = data8.iloc[[1],:]
print(mcoeff_data8.shape)
new = mcoeff_data8.stack().tolist()
coeffs_data8 = np.polyfit(K,new,9)

print(coeffs_data8)
p = np.poly1d(coeffs_data8)
plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Polynomial Fitting",fontsize = 12)
plt.plot(K,p(K))


# # Data 9

# In[223]:


mcoeff_data9 = data9.iloc[[1],:]
print(mcoeff_data9.shape)
new = mcoeff_data9.stack().tolist()
coeffs_data9 = np.polyfit(K,new,9)

print(coeffs_data9)
p = np.poly1d(coeffs_data9)
plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Polynomial Fitting",fontsize = 12)
plt.plot(K,p(K))


# # Data 10

# In[224]:


mcoeff_data10 = data10.iloc[[1],:]
print(mcoeff_data10.shape)
new = mcoeff_data10.stack().tolist()
coeffs_data10 = np.polyfit(K,new,9)

print(coeffs_data10)
p = np.poly1d(coeffs_data10)
plt.figure(figsize=(5,4))
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Polynomial Fitting",fontsize = 12)
plt.plot(K,p(K))


# ## Feature Matrix

# In[183]:


poly_mod_1 = poly_mod_1.T
print(poly_mod_1.shape)
print(mod_data6.shape)


# In[184]:


print(mod_data6.columns.values)


# In[185]:


feature_matrix = pd.DataFrame( np.concatenate( (mod_data6.values, poly_mod_1.values), axis=1 ) )


# In[186]:


feature_matrix.drop(feature_matrix.iloc[:, 0:33], inplace = True, axis = 1) 


# In[187]:


feature_matrix.fillna(feature_matrix.mean(), inplace = True)


# In[188]:


print(feature_matrix.columns.values)


# ### PCA

# In[193]:


columns = np.array(['mean', 'median', 'variance', 'SMA', 'EMA', 'FFT',               'Coeff_1', 'Coeff_2', 'Coeff_3', 'Coeff_4', 'Coeff_5',                'Coeff_6', 'Coeff_7', 'Coeff_8', 'Coeff_9', 'Coeff_10'])


# In[194]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
feature_matrix_scaled = scaler.fit_transform(feature_matrix)
feature_matrix_scaled = pd.DataFrame(feature_matrix_scaled,columns = columns)
feature_matrix_scaled.head()


# In[195]:


from sklearn.decomposition import PCA
pca = PCA()
eigen_vectors = pca.fit_transform(feature_matrix_scaled)
eigen_vectors = pd.DataFrame(eigen_vectors)
eigen_vectors


# In[196]:


principal_components = pd.DataFrame(pca.components_, columns=columns)
principal_components


# In[197]:




plt.xlabel("Principal Components")
plt.ylabel("Variance")

pcs = ['pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5', 'pc_6', 'pc_7', 'pc_8', 'pc_9', 'pc_10']

bar = plt.bar(pcs, pca.explained_variance_[:10])

for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, str(round(height, 2)), ha='center', va='bottom')

plt.show()


# In[198]:


plt.figure(figsize=(16, 4), dpi=150)

plt.xlabel("Principal Components")
plt.ylabel("Variance Ratio")

pcs = ['pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5', 'pc_6', 'pc_7', 'pc_8', 'pc_9', 'pc_10']

bar = plt.bar(pcs, pca.explained_variance_ratio_[:10])

for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, str(round(height, 2)), ha='center', va='bottom')

plt.show()


# In[199]:


df_pc1_val = np.flip(np.sort(principal_components.loc[0]))
df_pc1_idx = np.flip(np.argsort(principal_components.loc[0]))
df_pc1 = pd.DataFrame([df_pc1_val], columns=columns[df_pc1_idx])

df_pc2_val = np.flip(np.sort(principal_components.loc[1]))
df_pc2_idx = np.flip(np.argsort(principal_components.loc[1]))
df_pc2 = pd.DataFrame([df_pc2_val], columns=columns[df_pc2_idx])

df_pc3_val = np.flip(np.sort(principal_components.loc[2]))
df_pc3_idx = np.flip(np.argsort(principal_components.loc[2]))
df_pc3 = pd.DataFrame([df_pc3_val], columns=columns[df_pc3_idx])

df_pc4_val = np.flip(np.sort(principal_components.loc[3]))
df_pc4_idx = np.flip(np.argsort(principal_components.loc[3]))
df_pc4 = pd.DataFrame([df_pc4_val], columns=columns[df_pc4_idx])

df_pc5_val = np.flip(np.sort(principal_components.loc[4]))
df_pc5_idx = np.flip(np.argsort(principal_components.loc[4]))
df_pc5 = pd.DataFrame([df_pc5_val], columns=columns[df_pc5_idx])


# In[200]:


df_pc1


# In[201]:


df_pc2


# In[202]:


df_pc3


# In[203]:


df_pc4


# In[204]:


df_pc5


# In[205]:


np.sum(pca.explained_variance_ratio_[:5])


# In[206]:


pca.explained_variance_ratio_


# In[ ]:




