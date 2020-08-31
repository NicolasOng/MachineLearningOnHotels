import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('hotel_scores_manual.csv')
df = df.dropna()
df = df[df["is_hotel"] == "yes"]
df = df.drop(columns=['is_hotel'])

df.to_csv('hotel_scores.csv', index=False)

print(df)
print(df.columns)
