# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:05:22 2021

@author: arman
"""
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('properties.csv')

# ==============
# Data Cleaning
# ==============

#Since address is all none and we already know from the definition of the problem that every observation is from San Francisco, CA we might as well exclude all of this...
dataset=dataset.drop(['address','city','state'],axis=1)

#We rename 'facts and features column' to 'beds' and 'real estate provider' to 'provider'...
dataset = dataset.rename(columns={'facts and features': 'beds', 'real estate provider': 'provider'})
#We split the newly renamed 'beds' column into 'beds','baths','sq_feet'
dataset[['beds','baths','sq_feet']] = dataset.beds.str.split(",",expand=True)

#We organize the format of our dataset
dataset=dataset[['url','title','postal_code','provider','beds','baths','sq_feet','price']]
#If you want to keep the 'url' column for some reason, dont run this line of code. I will because I dont need this for further analysis
dataset=dataset.drop(['url'],axis=1)
#Check for nulls
dataset.isnull().sum()
dataset.info()
#'provider' have 186 null values, so we'll have to deal with that later...


#Im interested that 'beds','baths','sq_feet', and 'price' columns are type object, so I might as well see why...
print(dataset['title'].unique())
print(len(dataset['title'].unique()))

print(dataset['beds'].unique())
print(len(dataset['beds'].unique()))

print(dataset['baths'].unique())
print(len(dataset['baths'].unique()))

print(dataset['price'].unique())
print(len(dataset['price'].unique()))

#From that short analysis we observe 3 things
#1-. There are 'None' Values that represent 0 in both 'beds' and 'baths' columns
#2-.There are commas in the 'price' column
#3-. There are letter or special characters in all of the three columns, we only want numbers
#Let's see what we can do about this

#First, lets deal with point No.1 and replace 'bds' unit for blank spaces and 'None' for just 0
dataset['beds'] = dataset['beds'].str.replace('bds','')
dataset['beds'] = dataset['beds'].str.replace('None','0')

dataset['baths'] = dataset['baths'].str.replace('ba','')
dataset['baths'] = dataset['baths'].str.replace('None','0')

dataset['sq_feet'] = dataset['sq_feet'].str.replace('sqft','')
dataset['sq_feet'] = dataset['sq_feet'].str.replace('None','0')

#Let's see how it looks
dataset.head()
#Looks fine, so lets go with point  2 and 3 on the list, replacing the non-numeric characters in 'price'

dataset['price'] = dataset['price'].str.replace('+','')
dataset['price'] = dataset['price'].str.replace('$','')
dataset['price'] = dataset['price'].str.replace(',','')

dataset.head()

#Great, now lets just convert from type object to float, since we are interested in floats for further analysis and modeling
dataset["beds"] = pd.to_numeric(dataset["beds"], downcast="float")
dataset["baths"] = pd.to_numeric(dataset["baths"], downcast="float")
dataset["sq_feet"] = pd.to_numeric(dataset["sq_feet"], downcast="float")
dataset["price"] = pd.to_numeric(dataset["price"], downcast="float")
dataset.info()

print(dataset['title'].unique())
print(len(dataset['title'].unique()))

#Now, let's deal with the missing data.
#Since we want our model to be as accurate as possible, we'll drop the rows that have null values.
#We could use the mode, but that would mean suppositions from my part, and I want the model to be very accurate
#Also, 891 observations is plenty of data for this type of problem.
dataset = dataset[dataset['provider'].notna()]

#Now our dataset is ready for analysis and modeling

file_name="cleaned_properties.csv"
dataset.to_csv(file_name, sep=',', encoding='utf-8',index=False)