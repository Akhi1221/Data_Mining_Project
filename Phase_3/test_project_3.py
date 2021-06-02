# -*- coding: utf-8 -*-
"""
Created on Sun Mar  21 17:19:00 2020

@author: B511 I7
"""

import csv
import pandas as pd
import numpy as np
from numpy import array
from numpy import matrix
import statistics
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis as kurt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler as sc
import seaborn as sns; sns.set()
import itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial import distance
from sklearn import metrics


 ################## Feature Extraction from Test Data###############
filenames_Test = ["mealData3.csv"]
count=0
print("1")
featureMatrixTest =[]
for CGM in filenames_Test:
    CGMSeries5 = pd.read_csv(CGM,header =None)
    #print(CGMSeries5)
    a = list(CGMSeries5)
    #print(CGM)
    
    for i,row in CGMSeries5.iterrows():

        skew_value = skew(list(row))
       
        kurt_value = kurt(list(row))
        
       
        fft1 = abs(np.fft.fft(row))
        np.random.shuffle(fft1)
        top8fft = fft1[np.argsort(fft1)[-8:]]

        x_axis = []
        for j in range(1,len(CGMSeries5.columns)+1):
            x_axis.append(0.0035*j)
     
        polyfit_coeff = np.polyfit(x_axis,row[::-1],4)
    
        featureVector = [kurt_value,top8fft[0],top8fft[6],top8fft[7],polyfit_coeff[0],polyfit_coeff[3],polyfit_coeff[4]]
        #print(featureVector)
        if count==0:
            featureMatrixTest = featureVector
        else:
            featureMatrixTest = np.append(featureMatrixTest,featureVector)
        count = count + 1
    

split_list = np.split(featureMatrixTest,count)
featureMatrixTest = np.array(split_list)
normalized_feature_matrix_Test = sc().fit_transform(featureMatrixTest)

TestingMealData = np.append(featureMatrixTest,[])
split_list = np.split(TestingMealData,len(featureMatrixTest))
TestingMealData = np.array(split_list)

###################### Feature Extraction from TrainData ####################
print("2")
filenames_meal = ["mealData1.csv","mealData2.csv","mealData3.csv","mealData4.csv","mealData5.csv"]
#filenames_meal = ["mealData1.csv"]

featureMatrixMeal =[]
count=0
for CGM in filenames_meal:
    CGMSeries5 = pd.read_csv(CGM,header =None)
    a = list(CGMSeries5)
    #print(CGMSeries5)
    print(CGM)
    
    for i,row in CGMSeries5.iterrows():
        
        skew_value = skew(list(row))
        kurt_value = kurt(list(row))
        
       
        fft1 = abs(np.fft.fft(row))
        np.random.shuffle(fft1)
        top8fft = fft1[np.argsort(fft1)[-8:]]
  
        
        x_axis = []
        for j in range(1,len(CGMSeries5.columns)+1):
            x_axis.append(0.0035*j)
        #print(x_axis)
        #print(row[::-1])
        polyfit_coeff = np.polyfit(x_axis,row[::-1],4)
        
        ####################Considering 7 features for clustering#############
        featureVector = [kurt_value,top8fft[0],top8fft[6],top8fft[7],polyfit_coeff[0],polyfit_coeff[3],polyfit_coeff[4]]
    
        if count==0:
            featureMatrixMeal = featureVector
        else:
            featureMatrixMeal = np.append(featureMatrixMeal,featureVector)
        count = count + 1
        
split_list = np.split(featureMatrixMeal,count)
featureMatrixMeal = np.array(split_list)
normalized_feature_matrix_Meal = sc().fit_transform(featureMatrixMeal)

TrainingMealData = np.append(featureMatrixMeal,[])
split_list = np.split(TrainingMealData,len(featureMatrixMeal))
TrainingMealData = np.array(split_list)


######################## KMeans Clustering the Train Data ###################

clusterer = KMeans(n_clusters=6).fit(TrainingMealData)
c1=clusterer.labels_+1
Kmeans_centroid =clusterer.cluster_centers_[:,:]
print(Kmeans_centroid.shape)
print("\n clusters assiged using Kmeans to meal data : \n")
print(c1)
"""print("\n The centroids obtained for 6 clusters using Kmeans : \n")
print(Kmeans_centroid)"""
#print(clusterer.inertia_)

##################### Assigning labels to the test data(KMeans)######################

Data = TestingMealData
Centroid = Kmeans_centroid
Labels = [1,2,3,4,5,6]

Kmeans_labels = []
m = 0
for i in range(0,len(Data)):
    for j in range(0,len(Centroid)):
        p = m
        if distance.euclidean(Data[i],Centroid[p])<distance.euclidean(Data[i],Centroid[j]):
            q = p
            m = p
        else:
            q = j
            m = j
        if j==1:
            Kmeans_labels.append(q+1)

print("\n the Labels assigned to Test data using KMeans \n")
print(Kmeans_labels)
sse =clusterer.inertia_
print("\n Sum of squared error: \n")
print(sse/10**6)

a = Kmeans_labels
#################### Extracting Labels from Meal Amount Data #################
filenames_meal = ["mealAmountData1.csv","mealAmountData2.csv","mealAmountData3.csv","mealAmountData4.csv","mealAmountData5.csv"]
CarbMeal =[]

for CGM in filenames_meal:
    CGMSeries5 = pd.read_csv(CGM,header =None)
    CarbMeal = np.append(CarbMeal,CGMSeries5)

CarbMeal =CarbMeal.reshape(-1, 1)

df1 = pd.DataFrame(CarbMeal,columns=[0])
criteria = [ df1[0].between(0,0), df1[0].between(0,20),df1[0].between(20,40),df1[0].between(40,60),df1[0].between(60,80),df1[0].between(80,100)]
values = [1, 2, 3,4,5,6]
df1[1] = np.select(criteria, values, 0)
print("\n the labels using range for the meal amount data \n")
print(df1)
c6 = df1[1]

############################## DBSCAN #####################################
from sklearn.preprocessing import StandardScaler
alpha = [1]
beta = [2]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(featureMatrixMeal)
dbscan = DBSCAN(eps=1.1, min_samples = 2)
clusters = dbscan.fit_predict(X_scaled)
for i in alpha:
    for j in beta:
        
        
        clustering = DBSCAN(eps=i, min_samples=j).fit(X_scaled)
        c3 = clustering.labels_
        (unique, counts) = np.unique(c3, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        


count1 = 20
DBSCAN_labels = []
for i in range(0,len(c3)):
    if c3[i] == -1 :
        DBSCAN_labels.append(4)
    elif c3[i] == 0:
        DBSCAN_labels.append(1)
    elif 23<=c3[i]<=24:
        DBSCAN_labels.append(6)
    elif 18<=c3[i]<=22:
        DBSCAN_labels.append(5)
    elif 8<=c3[i]<=17:
        DBSCAN_labels.append(3)
    else:
        DBSCAN_labels.append(2)

print("\n clusters assiged using DBSCAN to meal data : \n")
print((DBSCAN_labels))
(unique, counts) = np.unique(DBSCAN_labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print("\n The distribution of the clusters obtained by DBSCAN : \n")
print(frequencies)        




############################## DBSCAN Error #################################


scaler = StandardScaler()
X_scaled = scaler.fit_transform(featureMatrixMeal)
X_test   = scaler.fit_transform(featureMatrixTest)
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_scaled, DBSCAN_labels)
y_train_pred = neigh.predict(X_scaled)
y_test_pred  = neigh.predict(X_test)
print("\n the Labels assigned to Test data using DBSCAN \n")
print(y_test_pred)
print(len(y_test_pred))
print(len(Kmeans_labels))
print("\n K = 7 is optimal from training ")
print("\ the output matrix containing both Kmeans and DBSCAN labels of test data")
column_names = ["Kmeans_labels","DBSCAN_labels"]
df10 = pd.DataFrame(columns = column_names)
df10["Kmeans_labels"] = Kmeans_labels
df10["DBSCAN_labels"] = y_test_pred
print(df10)
print("-"*60)
