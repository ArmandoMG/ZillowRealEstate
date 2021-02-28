# Real Estate Predictor

* Created a tool that can help you to estimate properties prices in all of San Francisco, CA.
* Optimized MLR, SVR, Decision Tree, and Random Forest Regressors using RandomizedSearchCV to reach the best model.
* Scraped 1000+ properties descriptions from Zillow.com using selenium and python
* Performed Data cleaning in the scraped data to prepare the dataset for analyzing and modeling
* Performed EDA on the cleaned data

**NOTE: FOR A BETTER EXPLANATION OF THE GRAPHS, CODE, AND LOGIC BEHIND MY REASONING, CHECK THE JUPYTER NOTEBOOKS IN THE REPO**

## Code and Resources used

**Python Version:** 3.8

**Spyder Version:** 4.1.4

**Packages:** sklearn, pandas, numpy, selenium, matplotlib, seaborn

**Scraper github:** https://gist.github.com/scrapehero/5f51f344d68cf2c022eb2d23a2f1cf95

## Web Scraping
 Modify the web scrapper (github reference above) to scrape 1000+ properties info from Zillow.com. Each propertie have the following features:
 * Title (What type of property is? House, Condo, New construction, Under Contract, etc)
 * City
 * State
 * Postal Code
 * Price
 * Facts and Features (Bedrooms, bathrooms and squared feets)
 * Real Estate Provider
 * URL
 
## Data Cleaning
Once the data was scraped, cleaning the data was needed so it was usable for the EDA and model building. The changes made are the following:
 * Removed city, state, URL, and address columns
 * Renamed Columns for easy tracking
 * Separated Facts and Features column into 3 different columns (beds, baths, sq_feet)
 * Dealt with missing data by removing it
 * Removed special characters in numeric data such as letters (bd, bt, sq), $, commas, etc
 * Convert numeric data into its corresponding type of data 
 * Removed 1 property that was an outlier with misleading information

## Exploratory Data Analysis (EDA)
After the data was cleaned, I performed a EDA to get a better understanding of the situation and my data. Below are some of the more significant information obtained from the EDA (for more info and graphs check the jupyter notebooks):

![Figure 2021-02-27 111909](https://user-images.githubusercontent.com/24629475/109428330-e4394f00-79b3-11eb-8f8d-927feea748f5.png)

![Figure 2021-02-27 111909 (1)](https://user-images.githubusercontent.com/24629475/109428307-ca980780-79b3-11eb-86d5-c8a9bd5e7fcc.png)

![image](https://user-images.githubusercontent.com/24629475/109428266-9290c480-79b3-11eb-8c47-bb754bc3a149.png)

![image](https://user-images.githubusercontent.com/24629475/109428370-25c9fa00-79b4-11eb-9d42-f556056710c9.png)


Its worth pointing out that thanks to the EDA I found one property that supposedly had 84 bedrooms, after double-checking that out with the URL I found out it was some type of misleading information so I scraped that out. Also, after the EDA I droped the Provider column because it wasn't needed for the model building process.

## Model Building
First, I separated the data into independendt and dependent variables (my features and my target). Once I did that I used Binary Encoding for the categorical data
After that, I proceded with the training and test data split with a size of 20%.

For this project I tried with 4 different algorithms adn evaluated each of them to see how well the performed with this dataset. For the evaluation I used cross validation with R² as a score and MAE (Mean Absolute Error)

The models I tried were:
* **Multiple Linear Regression:** Base Model
* **Support Vector Regression:** It scales relatively well with high-dimensionality and the risk of overfitting is less
* **Decision Tree Regression:** Easy to build and interpret it. It does well with sparse data
* **Random Forest Regression:** With the sparsity of the data, this was a goood choice to try

## Models Performance and Results
The Random forest was far better than the other 3 choices.

* **Multiple Linear Regression:**
R² (Adjusted): 55.95% 
Cross Validation Score (R²): 49.95%
MAE: 772907.08

![image](https://user-images.githubusercontent.com/24629475/109430454-ccb39380-79be-11eb-9b22-a6f611378cc7.png)

* **Support Vector Regression:**
R² (Adjusted): 67.17%
Cross Validation Score (R²): 56.66%
MAE: 696211.43

![image](https://user-images.githubusercontent.com/24629475/109430393-734b6480-79be-11eb-809a-db668e9fb71a.png)

* **Decision Tree Regression:**
R² (Adjusted): 75.55% 
Cross Validation Score (R²): 63.06
MAE: 459040.7235284946

![image](https://user-images.githubusercontent.com/24629475/109430435-b60d3c80-79be-11eb-99b6-576bda8ae86f.png)

* **Random Forest Regression:**
R² (Adjusted): 85.37% 
Cross Validation Score (R²): 74.35%
MAE: 438138.64

![image](https://user-images.githubusercontent.com/24629475/109429845-64af7e00-79bb-11eb-9e23-186caa5d7fc0.png)

![image](https://user-images.githubusercontent.com/24629475/109429920-bbb55300-79bb-11eb-8a5b-7af7ce6faec1.png)

