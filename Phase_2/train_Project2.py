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
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score
from scipy.stats import skew
from scipy.stats import kurtosis as kurt

################################### MealData #####################################

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
X_train, X_test, y_train, y_test = train_test_split(TrainingData, label, test_size=0.33, stratify=label)




print("Train and Test accuracies using KNN")
K = [1,3, 5,7,11,13, 15,17, 21, 31]
for i in K:
    
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)

    y_train_pred = neigh.predict(X_train)
    y_test_pred  = neigh.predict(X_test)

    print("K = " + str(i))
    print("Train Accuracy = " + str(metrics.accuracy_score(y_train, y_train_pred) * 100))
    print("Test Accuracy = " + str(metrics.accuracy_score(y_test, y_test_pred) * 100))
    print("-"*100)

print("K=5 is  the optimal case")


knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, TrainingData, label, cv=3, scoring='accuracy')
print(scores.mean())


print("k=1")
knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn, TrainingData, label, cv=2, scoring='accuracy')
print(scores.mean())
print("k=3")
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, TrainingData, label, cv=2, scoring='accuracy')
print(scores.mean())
print("k=5")
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, TrainingData, label, cv=2, scoring='accuracy')
print(scores.mean())
print("k=7")
knn = KNeighborsClassifier(n_neighbors=7)
scores = cross_val_score(knn, TrainingData, label, cv=2, scoring='accuracy')
print(scores.mean())
print("k=9")
knn = KNeighborsClassifier(n_neighbors=9)
scores = cross_val_score(knn, TrainingData, label, cv=2, scoring='accuracy')
print(scores.mean())
print("K=11")
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, TrainingData, label, cv=2, scoring='accuracy')
print(scores.mean())
print("k=13")
knn = KNeighborsClassifier(n_neighbors=13)
scores = cross_val_score(knn, TrainingData, label, cv=2, scoring='accuracy')
print(scores.mean())