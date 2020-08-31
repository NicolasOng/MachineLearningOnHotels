import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#create a distance function
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

def main():
	'''
	this python script takes in the provided amenities and tourism in vancouver JSON files,
	and outputs a csv file with each row representing a hotel, its location, and the amenities nearby.
	'''
	#1. get the dataframes of the input files
	amen_df = pd.read_json("amenities-vancouver.json.gz", lines=True)
	tour_df = pd.read_json("tourism-vancouver.json.gz", lines=True)
	buil_df = pd.read_json("buildings-vancouver.json.gz", lines=True)
	
	#2. define amenities typical hotel-goers care about, AND >40 in amenities file (32 total)
	desired_amen = ["bench", "restaurant", "bicycle_parking", "fast_food", "waste_basket", "cafe", "post_box", "toilets", "bank", "drinking_water", "pharmacy", "parking", "parking_entrance", "dentist", "fuel", "bicycle_rental", "pub", "post_office", "bar", "recycling", "vending_machine", "car_sharing", "clinic", "place_of_worship", "atm", "waste_disposal", "ice_cream", "library", "charging_station", "fountain", "theatre", "car_rental"]
	
	#3. from the tourism and building df, get the hotels, hostels, motels, and chalets,
	#	in the format: name, lat, lon
	hotel_df_temp = tour_df[tour_df['tourism'] == "hotel"][['name', 'lat', 'lon']]
	hostel_df_temp = tour_df[tour_df['tourism'] == "hostel"][['name', 'lat', 'lon']]
	motel_df_temp = tour_df[tour_df['tourism'] == "motel"][['name', 'lat', 'lon']]
	chalet_df_temp = tour_df[tour_df['tourism'] == "chalet"][['name', 'lat', 'lon']]
	building_df_temp = buil_df[buil_df['building'] == "hotel"][['name', 'lat', 'lon']]
	
	hotel_df = pd.concat([hotel_df_temp,
							hostel_df_temp,
							motel_df_temp,
							chalet_df_temp,
							building_df_temp])
	print(hotel_df)
	
	#3. for each amenity, find:
	#	1. the distance to the closest one, (amenity-closest)
	#	2. and how many are within walking distance. (500m). (amenity-near)
	for amen in desired_amen:
		print("finding features for", amen)
		#3.a) find both aspects for all hotels
		cl_ne_df = hotel_df.apply(amenity_access,
									amenity=amen,
									all_amenities=amen_df,
									axis=1,
									result_type='expand')
		#3.b) add the two columns to the hotel dataframe
		hotel_df = pd.concat([hotel_df, cl_ne_df], axis=1, sort=False)
		#3.c) rename the two columns
		hotel_df = hotel_df.rename(columns={0: amen + '-closest', 1: amen + '-near'})
	
	print(hotel_df)
	print(hotel_df.columns)
	
	#4. write the data to csv
	hotel_df.to_csv('hotel_data.csv', index=False)

if __name__ == '__main__':
	main()


