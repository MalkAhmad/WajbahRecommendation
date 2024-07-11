# -*- coding: utf-8 -*-

from surprise import AlgoBase, PredictionImpossible

class HybridAlgorithm(AlgoBase):

    def __init__(self, algorithms, weights, sim_options={}):
        AlgoBase.__init__(self)
        self.algorithms = algorithms
        self.weights = weights

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        for algorithm in self.algorithms:
            algorithm.fit(trainset)
                
        return self

    def estimate(self, u, i):
        
        sumScores = 0
        sumWeights = 0
        
        for idx in range(len(self.algorithms)):
            try:
                score = self.algorithms[idx].estimate(u, i)
                weight = self.weights[idx]
                sumScores += score * weight
                sumWeights += weight
            except PredictionImpossible:
                # If the algorithm cannot make a prediction, we skip it
                continue
            
        if sumWeights == 0:
            raise PredictionImpossible('All algorithms failed to provide a prediction.')
            
        return sumScores / sumWeights
