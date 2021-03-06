# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:04:46 2021

@author: arman
"""

import requests 
from data_input import input_data

URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}
data = {"input": input_data}

r = requests.get(URL,headers=headers, json=data) 

r.json()