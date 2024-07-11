import numpy as np
import pandas as pd
from surprise import AlgoBase, PredictionImpossible
from RBM import RBM  # Assuming RBM class is defined in RBM.py

class RBMAlgorithm(AlgoBase):

    def __init__(self, epochs=20, hiddenDim=100, learningRate=0.001, batchSize=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.batchSize = batchSize

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # Load ratings
        ratings = pd.read_csv('Wajbah_validation_data.csv')
        ratings.dropna(axis=0, inplace=True)

        # Create mappings for user and item IDs to indices
        unique_users = ratings['User_ID'].unique()
        unique_items = ratings['Food_ID'].unique()
        user_index = {user_id: index for index, user_id in enumerate(unique_users)}
        item_index = {item_id: index for index, item_id in enumerate(unique_items)}

        numUsers = len(unique_users)
        numItems = len(unique_items)

        # Initialize training matrix
        self.trainingMatrix = np.zeros([numUsers, numItems * 6], dtype=np.float32)  # Adjusted size to accommodate ratings 0-5

        for index, row in ratings.iterrows():
            uid = user_index[int(row['User_ID'])]
            iid = item_index[int(row['Food_ID'])]
            rating = int(row['Rating'])

            # Ensure the adjusted rating is within the bounds
            if 0 <= rating < 6:
                self.trainingMatrix[uid, iid * 6 + rating] = 1
            else:
                print(f"Adjusted rating out of bounds: {rating}")

        # Create an RBM with (num items * rating values) visible nodes
        rbm = RBM(self.trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim, learningRate=self.learningRate,
                  batchSize=self.batchSize, epochs=self.epochs)
        rbm.Train(self.trainingMatrix)

        self.predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)
        for uiid in range(numUsers):
            if uiid % 50 == 0:
                print("Processing user ", uiid)
            recs = rbm.GetRecommendations([self.trainingMatrix[uiid]])
            recs = np.reshape(recs, [numItems, 6])

            for itemID, rec in enumerate(recs):
                normalized = self.softmax(rec)
                rating = np.average(np.arange(6), weights=normalized)
                self.predictedRatings[uiid, itemID] = rating

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        rating = self.predictedRatings[u, i]

        if rating < 0.001:
            raise PredictionImpossible('No valid prediction exists.')

        return rating
