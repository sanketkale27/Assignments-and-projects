# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:49:26 2020

@author: sanky
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_regression

class KnnAlgo:
    def __init__(self,config):
        self.testData = np.genfromtxt("testData.csv",delimiter=",")
        self.trainingData1 = np.genfromtxt("trainingData.csv",delimiter=",")
        self.td = np.delete(self.testData, 12, 1)
        self.trainD = np.delete(self.trainingData1, 12, 1)       
        self.k = 3
        training_lable = self.trainingData1[:,12]
        self.config = config
        if self.config == "1":
            print()
            print("Standard KNN ")
              
        if self.config == "2": 
        #SelectionPercentile
            print()
            print("SelectionPercentile ")
            x = SelectPercentile(f_regression, percentile=50)
            self.trainD = x.fit_transform(self.trainD, training_lable)
            self.td = x.transform(self.td)
        elif self.config == "3":
        # Normalization 
            print()
            print("Normalization ")
            self.td = preprocessing.normalize(self.td, norm='l2')
            self.trainD = preprocessing.normalize(self.trainD, norm='l2')
        elif self.config == "4":
        #Scale Features 
            #MaxAbsScaler
            print()
            print("MaxAbsScaler ")
            maxabs = preprocessing.MaxAbsScaler()
            self.td = maxabs.fit_transform(self.td)
            self.trainD = maxabs.fit_transform(self.trainD)
        elif self.config == "5":
            #MinMaxScaler
            print()
            print("MinMaxScaler ")
            minmax = preprocessing.MinMaxScaler()
            self.td = minmax.fit_transform(self.td)
            self.trainD = minmax.fit_transform(self.trainD)
        elif self.config == "6":
        #     #StandardScaler
            print()
            print("StandardScaler ")
            scalerTD = preprocessing.StandardScaler().fit(self.td)
            scalerTD.transform(self.td)
            scalerTTD = preprocessing.StandardScaler().fit(self.trainD)
            scalerTTD.transform(self.trainD)
        
       
    def calculate_distance(self,testData,trainingData):   
        sum_sq = np.sqrt(np.sum(np.square(testData - trainingData),axis =1))   
        return sum_sq 


    def Predict(self,testData,trainingData):
       predictarr = []
       size =0
       while len(testData) > size:       
            caldist = [] 
            caldist = self.calculate_distance(testData[[size]],trainingData)
    
            sortedarray= np.array(np.sort(caldist))            
            nearestN = sortedarray[:self.k] 
            weight= 1/nearestN
            
            indexofdis =np.argsort(caldist)[:self.k]
    
            targetedvalue=self.trainingData1[indexofdis] [:,-1]
    
            predictvalue = ((np.sum((weight)*targetedvalue))/np.sum(weight))
            predictarr.append(predictvalue)
            size += 1
        
       return np.array(predictarr)



    def calculate_r2(self,calreg,targetreg):
        sumofsquResiduals = np.sum(np.square(calreg - targetreg))
        sumofsquare = np.sum(np.square(np.mean(targetreg)-targetreg))
        r2  = 1- (sumofsquResiduals/sumofsquare)       
        return r2

    
    def main(self):
        targetedreg = [self.Predict(self.td,self.trainD)]   
        r2 = self.calculate_r2(targetedreg, self.testData[:,-1])      
       
        regression_value=[]
        for k in range(3,10):
            self.k = k
          
            targetedreg = [self.Predict(self.td,self.trainD)]   
            r2 = self.calculate_r2(targetedreg, self.testData[:,-1])
           
            print(r2)
            regression_value.append(r2)
        plt.plot([k for k in range(3,10)] ,regression_value,marker = 'o')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('K - NN Model')
        plt.show()
        
        
        

config = input("Please Enter function number (Standard KNN --1 , SelectionPercentile -- 2,Normalization -- 3 , MaxAbsScaler --4 ,MinMaxScaler --5,StandardScaler --6 )  ") 

knn =  KnnAlgo(config)
knn.main()     
   


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    