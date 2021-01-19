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
    def __init__(self):
        self.testData = np.genfromtxt("testData.csv",delimiter=",")
        self.trainingData1 = np.genfromtxt("trainingData.csv",delimiter=",")
        self.td = np.delete(self.testData, 12, 1)
        self.trainD = np.delete(self.trainingData1, 12, 1)       
        self.k = 3
        training_lable = self.trainingData1[:,12]
       
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
        print(r2)
        
        
        
knn =  KnnAlgo()
knn.main()     
   


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    