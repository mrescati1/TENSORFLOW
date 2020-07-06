# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:53:16 2020

@author: micha
"""
import numpy as np # for multi-dimensional containers
import pandas as pd # for DataFrames
import itertools
from chord import Chord

data_url = 'https://shahinrostami.com/datasets/pokemon.csv'
data = pd.read_csv(data_url)
data = data.dropna()
data = list(itertools.chain.from_iterable((i, i[::-1]) for i in data.values))
matrix = pd.pivot_table(pd.DataFrame(data), index=0, columns=1, 
                        aggfunc="size", fill_value=0).values.tolist()
pd.DataFrame(matrix)