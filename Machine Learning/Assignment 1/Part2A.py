# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:23:19 2020

@author: sanky
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self):
        self.trainingdata = np.genfromtxt("data.csv",delimiter=",")

    def generate_centroids(self,trainingdata,k):       
        k = np.random.randint(len(trainingdata), size=k)  
        arr = trainingdata[k]  
        return arr
    
    def calculate_distance(self,testData,trainingData):   
        sum_sq = np.sqrt(np.sum(np.square(testData - trainingData),axis =1))   
        return sum_sq 
    
    def assign_centroids(self,trainingData,gencentroid):
        cnt = 0
        distance = []
        distnacelist = []
        mindistance = 0
        sum_sq = []
        while len(trainingData) > cnt:
            sum_sq = np.argmin(self.calculate_distance(gencentroid,trainingData[cnt]))
            distnacelist.append(sum_sq)
            cnt +=1
        
        return distnacelist
            
    def move_centroids(self,assigcent,trainingdata,gencent):
        cnt = 0
        finalarray = []        
        while len(gencent) > cnt:   
            final = np.mean(trainingdata[np.array(assigcent) == cnt],axis=0)       
            finalarray.append(final)
            cnt += 1
        return finalarray
    
    def restart_KMeans(self,trainingdata,number_centroid,number_iteration,number_restart):
        costlist = []       
        assigcent = []
        gencent = []
        finalcen = []

        for i in range(0,number_restart):
            gencent = np.array(self.generate_centroids(trainingdata, k=number_centroid))
            for j in range(0,number_iteration):
                assigcent = np.array(self.assign_centroids(trainingdata,gencent))
                gencent = np.array(self.move_centroids(assigcent,trainingdata,gencent))            
            cost = self.calculate_cost(trainingdata,assigcent,gencent)
            finalcen.append(gencent)        
            costlist.append(cost)            
        
        return  finalcen[np.argsort(costlist)[0]],costlist[np.argsort(costlist)[0]]
        
    def calculate_cost(self,trainingdata ,assigcent,movecentroid):
         c =(np.sum(np.square(self.calculate_distance(movecentroid[assigcent],trainingdata))))*(1/len(trainingdata))
         return c
        
    
    def main(self):     
        costarrlist = []
        for k in range(2,10):
            cluster,cost  = self.restart_KMeans(self.trainingdata,k,10,10)
            costarrlist.append(cost)
        print(costarrlist)
        plt.plot([k for k in range(2,10)] ,costarrlist,marker = 'o')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('k with the Elbow Method')
        plt.show()
    
kmean = KMeans()
kmean.main()