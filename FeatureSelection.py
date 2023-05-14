import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

df = pd.read_csv('D:/Higgs Prediction/atlas_higgs.csv')

X = df.iloc[:1000, :-1].values
y = df.iloc[:1000, -1].values


# Filter Method
from sklearn.feature_selection import mutual_info_classif

importances = mutual_info_classif(X, y)
feat_importances = pd.Series(importances, df.columns[:-1])
feat_importances.plot(kind = 'barh', color = 'teal')
plt.show()

# Fisher's Score
from skfeature.function.similarity_based import fisher_score

ranks = fisher_score.fisher_score(X, y)

feat_importance = pd.Series(ranks, df.columns[:-1])
feat_importance.plot(kind = 'barh', color = 'aquamarine')

# Random Forest Importance

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 300)
model.fit(X, y)

importances = model.feature_importances_

final_df = pd.DataFrame({"Features": pd.DataFrame(X).columns, "Importances": importances})
final_df.set_index('Importances')

final_df = final_df.sort_values('Importances')
final_df.plot.bar(color = 'crimson')



# random forest for feature importance on a classification problem
from matplotlib import pyplot

model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
'''
fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['Label'],
                   colorscale = 'Electric',
                   showscale = True,
                   cmin = -4000,
                   cmax = -100),
        dimensions = list([
            dict(range = [32000,227900],
                 constraintrange = [100000,150000],
                 label = "Weight", values = df['Weight']),
            dict(range = [0,700000],
                 label = 'PRI_jet_all_pt', values = df['PRI_jet_all_pt']),
            dict(tickvals = [0,0.5,1,2,3],
                 ticktext = ['A','AB','B','Y','Z'],
                 label = 'PRI_jet_subleading_phi', values = df['PRI_jet_subleading_phi']),
            dict(range = [-1,4],
                 tickvals = [0,1,2,3],
                 label = 'PRI_jet_subleading_eta', values = df['PRI_jet_subleading_eta']),
            dict(range = [134,3154],
                 visible = True,
                 label = 'PRI_jet_subleading_pt', values = df['PRI_jet_subleading_pt']),
            dict(range = [9,19984],
                 label = 'PRI_jet_leading_phi', values = df['PRI_jet_leading_phi']),
            dict(range = [49000,568000],
                 label = 'PRI_jet_leading_eta', values = df['PRI_jet_leading_eta'])])
    )
)
fig.show()
'''