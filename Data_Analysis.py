# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('D:/Higgs Prediction/atlas_higgs.csv')
df = df.drop(['EventId'], axis = 1)

print(df.columns)

# Correlation Heatmap
plt.figure(figsize=(34, 12))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

'''
sns.pairplot(df, hue="Label")

sns.set_style("ticks")
sns.pairplot(df,hue = 'Label',diag_kind = "kde",kind = "scatter",palette = "husl")
plt.show()
'''