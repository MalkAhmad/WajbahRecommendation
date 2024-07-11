import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Define method to load ratings data
def load_Server_data():
    response = requests.get('https://wajbah-api.azurewebsites.net/api/MenuItemAPI/GetAllRatings')
    data = response.json()

    if data['isSuccess']:
        ratings_list = data['result']
        ratings = pd.DataFrame(ratings_list)
        ratings.columns = ['User_ID', 'Food_ID', 'Rating']
        return ratings
    else:
        raise Exception("Failed to load ratings data from API")

# Compute popular items based on average ratings
def compute_popular_items(ratings, n=5):
    item_mean_ratings = ratings.groupby('Food_ID')['Rating'].mean()
    popular_items = item_mean_ratings.nlargest(n).index.tolist()
    return popular_items

# Define method to get top N recommendations for a user
def get_top_n_recommendations(user_id, n=5):
    # Load the ratings data
    ratings = load_Server_data()

    # Precompute popular items if not already done
    global popular_items
    if popular_items is None:
        popular_items = compute_popular_items(ratings)

    # Get items rated by the user
    user_rated_items = ratings[ratings['User_ID'] == user_id]['Food_ID'].tolist()
    all_items = ratings['Food_ID'].unique().tolist()
    items_to_predict = list(set(all_items) - set(user_rated_items))

    print(f"Items to predict (not rated by user {user_id}): {items_to_predict}")

    # Predict ratings for items not rated by the user
    predictions = [Hybrid.predict(user_id, item_id) for item_id in items_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n_recommendations = [int(prediction.iid) for prediction in predictions[:n]]

    print(f"Top {n} recommendations for user {user_id}: {top_n_recommendations}")

    return top_n_recommendations

# Load the trained model
with open('hybrid_model.pkl', 'rb') as model_file:
    Hybrid = pickle.load(model_file)

# Initialize popular_items as None
popular_items = None

# Define method to get details of recommended items
def get_item_details(item_ids):
    url = "https://wajbah-api.azurewebsites.net/api/MenuItemAPI/FiveMenuItems"
    payload = item_ids
    headers = {"Content-Type": "application/json"}

    print(f"Fetching details for item IDs: {payload}")

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch item details. Status code: {response.status_code}")
            print(response.text)
            return None

    except requests.exceptions.RequestException as e:
        print(f"Exception occurred: {e}")
        return None

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    top_5_recommendations = get_top_n_recommendations(user_id, n=5)

    recommended_items = get_item_details(top_5_recommendations)

    if recommended_items:
        return jsonify(recommended_items)
    else:
        return jsonify({"error": "Failed to fetch recommended items"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
