#'''
INPUTFILE = "amenities-vancouver.json.gz"
GIVENTAG = "cuisine"
MAINTAG = "amenity"
#'''

'''
INPUTFILE = "buildings-vancouver.json.gz"
GIVENTAG = "source"
MAINTAG = "building"
#'''

'''
INPUTFILE = "tourism-vancouver.json.gz"
GIVENTAG = "information"
MAINTAG = "tourism"
#'''

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

entries = pd.read_json(INPUTFILE, lines=True)
#entries = entries[entries['tags'] != {}]
print(entries)
print(entries.columns)

tags_list = []
values_list = []

def func_list(tag):
	for t in tag.keys():
		tags_list.append(t)

def getTagCount(tag):
	return tags_list.count(tag)


#get all unique tags and their count:
print("UNIQUE TAGS/COUNTS")
entries['tags'].apply(func_list)
tag_counts = pd.Series(data=tags_list).value_counts()
print(tag_counts)

#select those with enough entries to matter
print("TAGS W/ > 40 ENTRIES")
tag_counts = tag_counts[tag_counts > 40]
print(tag_counts)

#see all the values/counts for the given tag:
print("UNIQUE VALUES/COUNTS IN:", GIVENTAG)
def getValues(tag):
	try:
		values_list.append(tag[GIVENTAG])
	except:
		return

entries['tags'].apply(getValues)
print(pd.Series(data=values_list).value_counts())

print("UNIQUE VALUES/COUNTS FOR:", MAINTAG)
#check how many of each amenity there is:
temp = entries[MAINTAG].value_counts()
print(temp)
for i in range (len(temp)):
	print(temp.index[i], temp[i])

