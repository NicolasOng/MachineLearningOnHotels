import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import GridSearchCV

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler

#from sklearn.neighbors import KNeighborsRegressor #(50) 0.2
from sklearn.ensemble import RandomForestRegressor#(n_estimators=50, max_depth=4) 0.5
#from sklearn.linear_model import LinearRegression #-60
#from sklearn.linear_model import Ridge #0

from sklearn.tree import export_graphviz

#1. read the hotel and their scores from disk
hotels = pd.read_csv('hotel_scores.csv')

#2. split them into features and the label
X = hotels.drop(columns=['name', 'score'])
y = np.ravel(hotels[['score']])

#3. split THOSE into a training set and a validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=209020)
#X_train, X_test, y_train, y_test = train_test_split(X, y)

#4. build a model to predict the hotel's score based on nearby amenities
#model = RandomForestRegressor(n_estimators=50, max_depth=4, max_leaf_nodes=10, random_state=20)
model = RandomForestRegressor(n_estimators=2000, max_depth=4, max_leaf_nodes=10, max_features=6)

#5. train the model with the training data
model.fit(X_train, y_train)

#6. print the model's score for the training data and validation data
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
