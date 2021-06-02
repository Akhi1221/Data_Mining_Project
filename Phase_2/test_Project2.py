

import csv
import pandas as pd
import numpy as np
from numpy import array
from numpy import matrix
import statistics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler as sc
import seaborn as sns; sns.set()
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle
from sklearn.metrics import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score
from scipy.stats import skew
from scipy.stats import kurtosis as kurt

############################### Feature Extraction From Test Data ################

filenames_Test = ["TestData.csv"]

featureMatrixTest =[]
count=0
for CGM in filenames_Test:
    Glucose_Series = pd.read_csv(CGM,header =None)
    a = list(Glucose_Series)
    print(CGM)
    
    for i,row in Glucose_Series.iterrows():
        
        
        skew_value = skew(list(row))
        kurt_value = kurt(list(row))
        fft1 = abs(np.fft.fft(row))
        np.random.shuffle(fft1)
        top_fft_8 = fft1[np.argsort(fft1)[-8:]]
    
        x_axis = []
        for j in range(1,len(Glucose_Series.columns)+1):
            x_axis.append(0.0035*j)
        
        polyfit_coeff = np.polyfit(x_axis,row[::-1],4)
       
        meanCGM =[0.0]*5
        windowSize = 10
        startSample = 1
        endSample = startSample + windowSize - 1
        k=0
        while k<5 and endSample <= len(Glucose_Series.columns):
            meanCGM[k]= statistics.mean(row.iloc[startSample:endSample])
            startSample = startSample + int(windowSize/2)
            endSample = startSample + windowSize
            k= k+1


        featureVector = [skew_value,kurt_value,top_fft_8[0],top_fft_8[1],top_fft_8[2],top_fft_8[3],top_fft_8[4],top_fft_8[5],top_fft_8[6],top_fft_8[7],polyfit_coeff[0],polyfit_coeff[1],polyfit_coeff[2],polyfit_coeff[3],polyfit_coeff[4]]
        
        if count==0:
            featureMatrixTest = featureVector
        else:
            featureMatrixTest = np.append(featureMatrixTest,featureVector)
        count = count + 1
    

split_list = np.split(featureMatrixTest,count)

featureMatrixTest = np.array(split_list)
TestingMealData = sc().fit_transform(featureMatrixTest)


print("testingMeal")
print(TestingMealData.shape)

###################Feature Extraction from Meal & NoMeal data ################

filenames_meal = ["mealData1.csv","mealData2.csv","mealData3.csv","mealData4.csv","mealData5.csv"]
#filenames_meal = ["mealData1.csv"]

feature_Mat_Meal =[]
count=0
for CGM in filenames_meal:
    Glucose_Series = pd.read_csv(CGM,header =None)
    a = list(Glucose_Series)
    #print(Glucose_Series)
    print(CGM)
    
    for i,row in Glucose_Series.iterrows():
       
        skew_value = skew(list(row))
        kurt_value = kurt(list(row))
     
        fft1 = abs(np.fft.fft(row))
        np.random.shuffle(fft1)
        top_fft_8 = fft1[np.argsort(fft1)[-8:]]
   
        x_axis = []
        for j in range(1,len(Glucose_Series.columns)+1):
            x_axis.append(0.0035*j)
        
        polyfit_coeff = np.polyfit(x_axis,row[::-1],4)
       
      
    
        ######### Window Mean Calculation ####################
        meanCGM =[0.0]*5
        ##print(statistics.mean(row_0))
        windowSize = 10
        startSample = 1
        endSample = startSample + windowSize - 1
        k=0
        while k<5 and endSample <= len(Glucose_Series.columns):
            meanCGM[k]= statistics.mean(row.iloc[startSample:endSample])
            startSample = startSample + int(windowSize/2)
            endSample = startSample + windowSize
            k= k+1
    
        
        featureVector = [skew_value,kurt_value,top_fft_8[0],top_fft_8[1],top_fft_8[2],top_fft_8[3],top_fft_8[4],top_fft_8[5],top_fft_8[6],top_fft_8[7],polyfit_coeff[0],polyfit_coeff[1],polyfit_coeff[2],polyfit_coeff[3],polyfit_coeff[4]]
       
        if count==0:
            feature_Mat_Meal = featureVector
        else:
            feature_Mat_Meal = np.append(feature_Mat_Meal,featureVector)
        count = count + 1
    

split_list = np.split(feature_Mat_Meal,count)

feature_Mat_Meal = np.array(split_list)
print(feature_Mat_Meal.shape)



normalized_feature_matrix_Meal = sc().fit_transform(feature_Mat_Meal)
print(type(normalized_feature_matrix_Meal))
print(normalized_feature_matrix_Meal.shape)

############################ NoMeal Data #######################################

filenames_nomeal = ["Nomeal1.csv","Nomeal2.csv","Nomeal3.csv","Nomeal4.csv","Nomeal5.csv"]

feature_Mat_NOMeal =[]
count=0
for CGM in filenames_nomeal:
    Glucose_Series = pd.read_csv(CGM)
    a = list(Glucose_Series)
    print(CGM)
    
    for i,row in Glucose_Series.iterrows():
        
        skew_value = skew(list(row))
        kurt_value = kurt(list(row))
        
        fft1 = abs(np.fft.fft(row))
        np.random.shuffle(fft1)
        top_fft_8 = fft1[np.argsort(fft1)[-8:]]
    

        x_axis = []
        for j in range(1,len(Glucose_Series.columns)+1):
            x_axis.append(0.0035*j)
        
        polyfit_coeff = np.polyfit(x_axis,row[::-1],4)
        meanCGM =[0.0]*5
        
        windowSize = 10
        startSample = 1
        endSample = startSample + windowSize - 1
        k=0
        while k<5 and endSample <= len(Glucose_Series.columns):
            meanCGM[k]= statistics.mean(row.iloc[startSample:endSample])
            startSample = startSample + int(windowSize/2)
            endSample = startSample + windowSize
            k= k+1

        featureVector = [skew_value,kurt_value,top_fft_8[0],top_fft_8[1],top_fft_8[2],top_fft_8[3],top_fft_8[4],top_fft_8[5],top_fft_8[6],top_fft_8[7],polyfit_coeff[0],polyfit_coeff[1],polyfit_coeff[2],polyfit_coeff[3],polyfit_coeff[4]]
        
        if count==0:
            feature_Mat_NOMeal = featureVector
            
        else:
            feature_Mat_NOMeal = np.append(feature_Mat_NOMeal,featureVector)
            
        count = count + 1
    

split_list = np.split(feature_Mat_NOMeal,count)

feature_Mat_NOMeal = np.array(split_list)
print(feature_Mat_NOMeal.shape)

normalized_feature_matrix_NOMeal = sc().fit_transform(feature_Mat_NOMeal)
print(type(normalized_feature_matrix_NOMeal))
print(normalized_feature_matrix_NOMeal.shape)
############################### Testing ####################################

TrainingData = np.vstack((normalized_feature_matrix_Meal, normalized_feature_matrix_NOMeal))
print("Training Data")
print(TrainingData.shape)
l2=len(normalized_feature_matrix_Meal)
labelMeal = np.ones((l2, 1), dtype =int)

l1=len(normalized_feature_matrix_NOMeal)
labelNOMeal = np.zeros((l1, 1), dtype =int)

label = np.append(labelMeal, labelNOMeal)

print(label.shape)
print(type(label))
print(type(TrainingData))
label = label.tolist()
print(type(label))

K = [5]
for i in K:
    
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(TrainingData, label)

    y_train_pred = neigh.predict(TrainingData)
    y_test_pred  = neigh.predict(TestingMealData)

    print("K = " + str(i))
    print("Train Accuracy = " + str(metrics.accuracy_score(label, y_train_pred) * 100))
    
    print("-"*100)

print(type(y_test_pred))
print(len(y_test_pred))

df = pd.DataFrame(y_test_pred)
df.to_csv('myfile.csv',index=False)





