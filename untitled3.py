# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:03:39 2020

@author: micha
"""


import numpy as np # for multi-dimensional containers
import pandas as pd # for DataFrames
import itertools
from chord import Chord

data_url = 'https://shahinrostami.com/datasets/pokemon_gen_1_to_8.csv'
data = pd.read_csv(data_url)
data = data.dropna()
types = pd.DataFrame(data[['type_1', 'type_2']].values)
types = list(itertools.chain.from_iterable((i, i[::-1]) for i in types.values))
matrix = pd.pivot_table(pd.DataFrame(types), index=0, columns=1, 
                        aggfunc="size", fill_value=0).values.tolist()
names = np.unique(type).tolist()
pd.DataFrame(matrix)
details = np.empty((len(names),len(names)),dtype=object)
for count_x, item_x in enumerate(names):
  for count_y, item_y in enumerate(names):
    details[count_x][count_y] = data[(data['type_1'].isin([item_x, item_y])) & (data['type_2'].isin([item_y, item_x]))]['name'].to_list()
details=pd.DataFrame(details).values.tolist()
Chord(matrix, names, details=details).show()
details = np.empty((len(names),len(names)),dtype=object)
for count_x, item_x in enumerate(names):
  for count_y, item_y in enumerate(names):
    details[count_x][count_y] = data[(data['type_1'].isin([item_x, item_y])) & (data['type_2'].isin([item_y, item_x]))]['name'].to_list()
details=pd.DataFrame(details).values.tolist()
Chord()