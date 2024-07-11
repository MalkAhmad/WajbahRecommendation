# -*- coding: utf-8 -*-


from surprise import AlgoBase
from surprise import PredictionImpossible
import math
import numpy as np
import heapq
import pandas as pd

class ContentKNNAlgorithm(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # Compute item similarity matrix based on content attributes
        print("Computing content-based similarity matrix...")
            
        # Load your ratings data from ratings.csv
        ratings = pd.read_csv('ratings.csv')
        ratings.dropna(axis=0, inplace=True)
        
        # Create a dictionary to store user-item ratings
        self.user_ratings = {}
        for _, row in ratings.iterrows():
            user = row['User_ID']
            food_id = row['Food_ID']
            rating = row['Rating']
            if user not in self.user_ratings:
                self.user_ratings[user] = {}
            self.user_ratings[user][food_id] = rating
        
        # Compute similarity matrix based on ratings data
        self.similarities = np.zeros((trainset.n_items, trainset.n_items))
        
        for thisRating in range(trainset.n_items):
            if (thisRating % 100 == 0):
                print(thisRating, " of ", trainset.n_items)
            for otherRating in range(thisRating+1, trainset.n_items):
                thisFoodID = int(trainset.to_raw_iid(thisRating))
                otherFoodID = int(trainset.to_raw_iid(otherRating))
                similarity = self.computeSimilarity(thisFoodID, otherFoodID)
                self.similarities[thisRating, otherRating] = similarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
                
        print("...done.")
                
        return self
    
    def computeSimilarity(self, food1, food2):
        # Function to compute similarity between two foods based on user ratings
        sumxx, sumxy, sumyy = 0, 0, 0
        for user in self.user_ratings:
            if food1 in self.user_ratings[user] and food2 in self.user_ratings[user]:
                x = self.user_ratings[user][food1]
                y = self.user_ratings[user][food2]
                sumxx += x * x
                sumyy += y * y
                sumxy += x * y
        
        if sumxx == 0 or sumyy == 0:
            return 0
        
        return sumxy / math.sqrt(sumxx * sumyy)

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
    
    # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
           food_id = rating[0]
           if food_id != i:  # Exclude the item we are trying to predict
              similarity = self.similarities[i, food_id]
              neighbors.append((similarity, rating[1]))
    
    # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
    
        if not k_neighbors:
            raise PredictionImpossible('No neighbors')
    
    # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
           if simScore > 0:
               simTotal += simScore
               weightedSum += simScore * rating
        
        if simTotal == 0:
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
