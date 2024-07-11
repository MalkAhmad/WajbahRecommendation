# WajbahRecommender
This is the Recommendation System used in Wajbah-User Application and Website


# 1. Tools and Integrated Development Environment (IDE)
## IDE:
**Visual Studio Code:** An extensible code editor with support for Dart and a lot of extensions for easier development.
**Google Colab:** A cloud-based Jupyter Notebook environment that allows for interactive coding and data analysis, we used for testing the model

## Tools:
**ngrok:** A tool used to deploy our Flask app and access local-hosted servers from outside the machine without a live server.
**Github:** A website and cloud-based service that helps developers store, manage, and collaborate on their code.
**Anaconda:** A distribution of Python providing an environment for developing and package management.


# 2. DataSet 
**Food.com Recipes and Interactions:** consists of 180K+ recipes and 700K+ recipe reviews covering 18 years of user interactions and uploads on “Food.com”. 

**Data Architecture:** The data includes three categories of files: raw files, preprocessed files, and split interaction files. We specifically utilize the split interaction files to align with our application data. These interaction files which contain 700K+ reviews are further divided into train, test, and validation sets. This is a sample of the train interaction file: 



**Preprocessing:** We have read the train test validation sets to preprocess over them and we find that the only compatible data will be {“ user_id”,” recipe_id”,” rating”}. Then we separated them and renamed them to be the following:

So we have 3 sets of the data 698k row, 7k row, 12.4k row.

# 3. Environment Setup 
## Required Packages and Libraries:
**numpy:** Provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on them.
pandas: Offers data structures and functions needed for data manipulation and analysis.
**Flask:** A  web framework for building web applications and APIs.
**scikit-surprise:** A Python library for building and analyzing recommender systems.
**flask-cors:** Enables Cross-Origin Resource Sharing (CORS) support for Flask applications, allowing cross-domain requests.
**requests:** Simplifies making HTTP requests in Python.
**tensorflow:** An open-source library for machine learning and ANN.




## Setup Instructions 
Download VS Code and Anaconda 
Create a new Python file and use the anaconda as the kernel 
Open a new terminal and write the following instructions:

NumPy: conda install numpy
pandas: conda install pandas
Flask: conda install flask
scikit-surprise: conda install -c conda-forge scikit-surprise
flask-cors: conda install flask-cors
requests: conda install requests
TensorFlow: conda install tensorflow

# 4. Evaluation metric
We have selected Mean Absolute Error (MAE) as our primary evaluation metric. This choice is driven by MAE's effectiveness in measuring the accuracy of our predictions, particularly in scenarios involving multiple algorithms.

MAE is calculated as the average of the absolute differences between predicted and actual ratings.


# 5. Machine Learning Algorithm 
**Hybrid Recommendation System:** Our hybrid approach combines two collaborative filtering methods, leveraging the strengths of Restricted Boltzmann Machines (RBM) and KNNBasic.
## Model Selection:
**RBM:** Powerful neural networks effective in capturing latent features and interactions in recommendation systems.
**KNNBasic:** An effective algorithm operates by finding the most similar items or users to a given target, based on cosine similarity.

## Hybridization:
**AlgoBase:** It takes multiple algorithms (RBM and KNNBasic in our case) and combines their predictions. We specify the weight of each algorithm which determines the contribution percentage of each model’s prediction in the final recommendation.


# 6. Results 
After training our model we managed to achieve Mean Absolute Error (MAE) score of 0.42 and a Root Mean Squared Error (RMSE) score of 1.04 on over 698k test cases


# 7. The Flask App Pipeline 
**Fetching user interaction data from the API**
First, we fetch all the necessary data for our model from the API, specifically the GetAllRatings endpoint, which provides user IDs, food item IDs, and the ratings each user gave

**Computing popular items**
Next, we compute the most popular items based on the highest average ratings across all data. These popular items will be recommended to new users or users with no past interactions.

**Provide the TopN Recommendations**
In this step, we take the user_id and compute the top N recommendations (N = 5 in our case). This returns the IDs of the 5 recommended food items.

**Fetch the Item details**
Finally, we fetch the details of the recommended items using their IDs. We send the list of 5 IDs to the FiveMenuItems endpoint, which provides all the attributes of the corresponding items.
