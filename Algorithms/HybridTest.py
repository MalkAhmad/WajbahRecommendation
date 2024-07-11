import pickle
from surprise import Dataset, Reader, KNNBasic
from surprise import accuracy

import pandas as pd
import random
import numpy as np
from RBMAlgorithm import RBMAlgorithm
from HybridAlgorithm import HybridAlgorithm

def LoadRatingsData(filename):
    ratings = pd.read_csv(filename)
    ratings.dropna(axis=0, inplace=True)  # Drop rows with missing values

    ratings['User_ID'] = ratings['User_ID'].astype(int)
    ratings['Food_ID'] = ratings['Food_ID'].astype(int)
    ratings['Rating'] = ratings['Rating'].astype(float)

    # Relaxed filtering: only filter users and items with more than 0 ratings
    user_rating_counts = ratings['User_ID'].value_counts()
    item_rating_counts = ratings['Food_ID'].value_counts()
    ratings = ratings[ratings['User_ID'].isin(user_rating_counts[user_rating_counts > 0].index)]
    ratings = ratings[ratings['Food_ID'].isin(item_rating_counts[item_rating_counts > 0].index)]

    return ratings

def LoadFoodData():
    return None

np.random.seed(0)
random.seed(0)

# Load training and test data
train_ratings = LoadRatingsData('Wajbah_validation_data.csv')
test_ratings = LoadRatingsData('Wajbah_train_data.csv')
food_data = LoadFoodData()

# Debugging statements to ensure data is loaded correctly
print("Train ratings sample:")
print(train_ratings.head())
print("Test ratings sample:")
print(test_ratings.head())

# Ensure there's enough data to split
if len(train_ratings) < 1 or len(test_ratings) < 1:
    raise ValueError("Not enough ratings in train or test data. Please provide more data.")

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(train_ratings[['User_ID', 'Food_ID', 'Rating']], reader)
trainset = train_data.build_full_trainset()

# Debugging statement to ensure trainset is created correctly
print("Number of ratings in trainset:", trainset.n_ratings)

test_data = Dataset.load_from_df(test_ratings[['User_ID', 'Food_ID', 'Rating']], reader)
testset = test_data.construct_testset(test_data.raw_ratings)

# Debugging statement to ensure testset is created correctly
print("Number of ratings in testset:", len(testset))

# Load the previously saved model if it exists
try:
    with open('hybrid_model.pkl', 'rb') as model_file:
        Hybrid = pickle.load(model_file)
    print("Loaded the existing model successfully.")
except FileNotFoundError:
    print("No existing model found. Creating a new one.")

    KNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    # Simple RBM
    SimpleRBM = RBMAlgorithm(epochs=20)
    # Collaborative Filtering using KNNBasic
    
    # Combine them in a Hybrid Algorithm
    Hybrid = HybridAlgorithm([KNN , SimpleRBM], [0.5, 0.5])

# Continue training the hybrid algorithm
#Hybrid.fit(trainset)

# Evaluate the model
predictions = Hybrid.test(testset)

# Compute and print RMSE and MAE
print("RMSE: ", accuracy.rmse(predictions))
print("MAE: ", accuracy.mae(predictions))

# Save the updated model
with open('hybrid_model.pkl', 'wb') as model_file:
    pickle.dump(Hybrid, model_file)

print("Model trained, evaluated, and saved successfully!")
