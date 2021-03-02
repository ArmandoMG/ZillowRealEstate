# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:07:12 2021

@author: arman
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('cleaned_properties.csv')

# ==============
# EDA
# ==============

#Lets start by looking our dataset
dataset.info()
dataset.head()

#A basic description of the dataset (just the numeric values)
dataset.describe()

#Check for null values
dataset.isnull().sum()

#Normal distribution of the price(target variable)
sns.distplot(dataset['price'])

#Distribution of Price(Transformed value on log scale)
fig, axarr = plt.subplots(figsize=(12, 10))
sns.distplot(np.log(dataset['price']),color='red',hist_kws=dict(edgecolor="k", linewidth=2)).set_title('Distribution of Price(Transformed value on log scale)')

#Heatmap of our numeric variables
fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(dataset.corr(),annot=True)

#Pairplot, compare this to the heatmap
sns.pairplot(dataset)

#Histograms of all numerical data
dataset.hist(bins=25, figsize=(25, 15), alpha=0.7, rwidth=0.80, layout=(2, 4));
#dataset['beds'].hist(bins=30,range=[0.1,10.0],).set_title('beds') borrar esto, pero si funciona, no se te olvide

#postal_code vs price (mean)
dataset.groupby('postal_code')['price'].mean()

#title_code vs price(mean)
dataset.groupby('title')['price'].mean()

#Boxplots of beds and baths vs price
fig, axarr = plt.subplots(2,1, figsize=(12,10),squeeze=False)

sns.boxplot(x=dataset['beds'],y=dataset['price'],ax=axarr[0][0]).set_title('Price with respect to number of bedrooms')
sns.boxplot(x=dataset['baths'],y=dataset['price'],ax=axarr[1][0]).set_title('Price with respect to no. of baths')

plt.subplots_adjust(hspace=.4)
sns.despine()

#https://www.zillow.com/homedetails/26-Varennes-St-San-Francisco-CA-94133/2073856389_zpid/
#es la casa de 84beds
#dataset=dataset[dataset.beds != 84]
#correr los boxplot otra vez

#pie chart for 'title' column
plt.pie(dataset.title.value_counts(), labels=dataset['title'].unique(), autopct='%1.1f%%', shadow=True, startangle=140)
dataset.title.value_counts()

plt.axis('equal')
plt.show()

#number of properties per postal code (which postal code have more properties for sale)
dataset.postal_code.value_counts()
print(len(dataset['postal_code'].unique()))

#number of properties per title (how many X type propertie is for sale right now?)
dataset.title.value_counts()
print(len(dataset['title'].unique()))

#number of providers selling properties (how many providers are selling right know in SF)
dataset.provider.value_counts().head()
print(len(dataset['provider'].unique()))


#Since provider does not provide anything to the modeling, we should exclude it before the modeling
dataset=dataset.drop(['provider'],axis=1)

file_name="modeling_properties.csv"
dataset.to_csv(file_name, sep=',', encoding='utf-8',index=False)
