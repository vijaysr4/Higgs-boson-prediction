# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing datasetss
df = pd.read_csv('D:/Higgs Prediction/atlas-higgs.csv')

df = df.drop(['KaggleSet', 'KaggleWeight'], axis = 1)
columns = list(df.head(0))
print(columns)

df['Label'] = df['Label'].replace({'s': '1'})
df['Label'] = df['Label'].replace({'b': '0'})
df.to_csv("D:/Higgs Prediction/atlas_higgs.csv", index = False)