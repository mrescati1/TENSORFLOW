# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:34:50 2020

@author: micha
"""
from chord import Chord
matrix = [ [0, 5, 6, 4, 7, 4], [5, 0, 5, 4, 6, 5], 
          [6, 5, 0, 4, 5, 5], [4, 4, 4, 0, 5, 5],
          [7, 6, 5, 5, 0, 4], [4, 5, 5, 5, 4, 0]]
names = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Thriller"]
Chord(matrix, names).show()