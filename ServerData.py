import pandas as pd
import requests
from surprise import Dataset, Reader
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# Define method to load ratings data
def load_Server_data():
    response = requests.get('https://wajbah-api.azurewebsites.net/api/MenuItemAPI/GetAllRatings')
    data = response.json()

    if data['isSuccess']:
        ratings_list = data['result']
        ratings = pd.DataFrame(ratings_list)
        ratings.columns = ['User_ID', 'Food_ID', 'Rating']
        ratings['User_ID'] = ratings['User_ID'].astype(int)
        ratings['Food_ID'] = ratings['Food_ID'].astype(int)
        return ratings
    else:
        raise Exception("Failed to load ratings data from API")
    
print(load_Server_data())