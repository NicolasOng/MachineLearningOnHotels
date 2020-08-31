# Hotels Analysis Project

This was a project created for a class at SFU. This was a two-person project, but I only uploaded my portion of it.

## Required Libraries

1. pandas
2. numpy
3. matplotlib
4. scipy
5. pyspark
6. lxml
7. requests
8. sklearn
9. plotly
10. sys
11. ast
12. scikit_posthocs

## Commands, arguments, and expected files

### Hotel Analysis

#### Quick

1. `python3 0-exploring-tags.py`
2. `spark-submit 1-osm-tourism.py /courses/datasets/openstreetmaps tourism`
3. `spark-submit 2-just-vancouver.py tourism tourism-vancouver`
4. `hdfs dfs -cat tourism-vancouver/* | gzip -d - | gzip -c > tourism-vancouver.json.gz`
	- Produces `tourism-vancouver.json.gz`
5. Repeat steps 2 & 3, for the `building` tag
	- Produces `buildings-vancouver.json.gz`
6. `python3 3-hotel-features.py`
	- Produces `hotel_data.csv`
7. Manually enter the scores for these hotels into `hotel_scores_manual.csv`
8. `4-hotel-score-clean.py`
	- Produces `hotel_scores.csv`
9. `5-hotel-regression.py`
10. `6-results-viz.py`
	- Produces `feature_importances.png` and `tree-x.dot`, x in [0, 20, 40, 60, 80]

#### Long

1. `python3 0-exploring-tags.py`
	- This prints information about the given OSM data. (`amenities-vancouver.json.gz`).
	- This file can be configured to print information about `buildings-vancouver.json.gz` and `tourism-vancouver.json.gz` by changing the variables at the top of the file.
2. `spark-submit 1-osm-tourism.py /courses/datasets/openstreetmaps tourism`
	- Run this on the spark cluster.
	- Takes about an hour.
	- This gets all the objects in OpenStreetMap with a `tourism` tag, and formats them into json.
	- Adapted from the given `osm-amenities.py`.
3. `spark-submit 2-just-vancouver.py tourism tourism-vancouver`
	- Run this on the spark cluster.
	- This gets all the objects from the previous step that are in Vancouver.
	- Adapted from the given `just-vancouver.py`.
4. `hdfs dfs -cat tourism-vancouver/* | gzip -d - | gzip -c > tourism-vancouver.json.gz`
	- This creates the file `tourism-vancouver.json.gz`, with the data from the previous step.
5. Repeat steps 2 & 3, but for objects in OpenStreetMap with a `building` tag.
	- This step produces `buildings-vancouver.json.gz`.
	-  Both `tourism-vancouver.json.gz` and `buildings-vancouver.json.gz` are given, so you don't have to wait hours for them.
6. `python3 3-hotel-features.py`
	- Creates the file `hotel_data.csv`, where each row is a hotel, and each column is either the distance from the nearest amenity or the amount of amenities within walking distance.
	- Results in 120 hotels.
7. Manually enter the scores for these hotels, takes about an hour. Many hotels have no name, or are just random houses. Produces `hotel_scores_manual.csv`.
8. `4-hotel-score-clean.py`
	- Produces `hotel_scores.csv` by cleaning the manual hotel file, by removing hotels that are found to be not hotels.
9. `5-hotel-regression.py`
	- Trains a model with the data in `hotel_scores.csv`, and prints the results.
10. `6-results-viz.py`
	- Gets how important the model thinks each feature is, and plots that in `feature_importances.png`.
	- Visualises the decision trees in the model, and puts that information into `tree-x.dot`, where x is 0, 20, 40, 60, 80.