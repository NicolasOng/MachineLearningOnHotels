import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz

import plotly.express as px
import plotly.graph_objects as go

#0. To start, import everything from previous setps needed for this one.
#this file really starts at 1
desired_amen = ["bench", "restaurant", "bicycle_parking", "fast_food", "waste_basket", "cafe", "post_box", "toilets", "bank", "drinking_water", "pharmacy", "parking", "parking_entrance", "dentist", "fuel", "bicycle_rental", "pub", "post_office", "bar", "recycling", "vending_machine", "car_sharing", "clinic", "place_of_worship", "atm", "waste_disposal", "ice_cream", "library", "charging_station", "fountain", "theatre", "car_rental"]
amen_df = pd.read_json("amenities-vancouver.json.gz", lines=True)

from math import cos, asin, sqrt, pi
def ptdistance(lat1, lon1, lat2, lon2):
	#from: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
	#in KM
    p = pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p) * np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2
    return 12742 * np.arcsin(np.sqrt(a))

def amenity_access(hotel_row, amenity, all_amenities):
	'''
	finds the distance of the closest amenity to hotel in the given row.
	'''
	#1. filter the amenities for the selected one
	selected_amenities = all_amenities[all_amenities['amenity'] == amenity]
	#2. find the distance of all selected amenities to the hotel
	dists = ptdistance(hotel_row['lat'],
						hotel_row['lon'],
						selected_amenities['lat'],
						selected_amenities['lon'])
	#3. return the distance of the closest one, and the amount of amenities within walking distance:
	return [dists.min(), len(dists[dists < 0.5])]

hotels = pd.read_csv('hotel_scores.csv')
X = hotels.drop(columns=['name', 'score'])
y = np.ravel(hotels[['score']])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=209020)
model = RandomForestRegressor(n_estimators=100, max_depth=4, max_leaf_nodes=100, max_features=6, random_state=50)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

#1. create a figure displaying the features the model deemed important
n_features = X.shape[1]
li = []
for i in range(n_features):
	if(i % 2 == 0):
		li.append("#e36387")
	else:
		li.append("#a6dcef")
plt.figure(figsize=[20, 15])
plt.barh(range(n_features), model.feature_importances_, align='center', color=li)
plt.yticks(np.arange(n_features), X.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.savefig("feature_importances.png", dpi=200)

#2. show the decision path of some trees to analyze
#NOTE: go to https://dreampuf.github.io/GraphvizOnline/ to see the files.
for i in range(0, 100, 20):
	estimator = model.estimators_[i]
	export_graphviz(estimator, out_file='tree-' + str(i) + '.dot', feature_names = X.columns, rounded = True, proportion = False, precision = 2, filled = True)

#3. display the hotels on a map
fig = px.scatter_mapbox(hotels, lat="lat", lon="lon", hover_name="name", color_discrete_sequence=["fuchsia"], zoom=3, height=1000)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

#4. display a heatmap of model-predicted scores on a map
las = []
los = []
ss = []
for lo in np.arange(-123.5, -122, ((-122)-(-123.5))/20):
	for la in np.arange(49, 49.5, (49.5-49)/10):
		las.append(la)
		los.append(lo)
		#find the estimated score
		hotel_df = pd.DataFrame(data={'name':[''], 'lat':[la], 'lon':[lo]})
		for amen in desired_amen:
			cl_ne_df = hotel_df.apply(amenity_access,
										amenity=amen,
										all_amenities=amen_df,
										axis=1,
										result_type='expand')
			#3.b) add the two columns to the hotel dataframe
			hotel_df = pd.concat([hotel_df, cl_ne_df], axis=1, sort=False)
			#3.c) rename the two columns
			hotel_df = hotel_df.rename(columns={0: amen + '-closest', 1: amen + '-near'})
		s = model.predict(hotel_df.drop(columns=['name']))[0]
		print(la, lo, s)
		ss.append(s)

fig = go.Figure(go.Densitymapbox(lat=las, lon=los, z=ss, radius=100))
fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=180)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

