# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:02:06 2020

@author: micha
"""


import numpy as np # for multi-dimensional containers
import pandas as pd # for DataFrames
import plotly.graph_objects as go # for data visualisation
import plotly.io as pio # to set shahin plot layout
from plotly.subplots import make_subplots
pio.templates['shahin'] = pio.to_templated(go.Figure().update_layout(
    legend=dict(orientation="h",y=1.1, x=.5, xanchor='center'),
    margin=dict(t=0,r=0,b=0,l=0))).layout.template
pio.templates.default = 'shahin'
data = pd.read_excel('https://www.arcgis.com/sharing/rest/content/items/e5fd11150d274bebaaf8fe2a7a2bda11/data')
data.head()
from plotly.subplots import make_subplots
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=data.DateVal, y=data.CumCases,
                         mode='lines+markers',name='Total Cases',
                         line_color='crimson'),secondary_y=True)
fig.add_trace(go.Bar(x=data.DateVal, y=data.CMODateCount, name='New Cases', 
                     marker_color='darkslategray'), secondary_y=False)
fig.show()