# AMES Housing Price Prediction

The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 

This project was started as a motivation for learning Machine Learning Algorithms and to learn the different data preprocessing techniques such as Exploratory Data Analysis, Feature Engineering, Feature Selection, Feature Scaling and finally to build a machine learning model. We will predicts house price in boston city

## DATA DESCRIPTION

The data was originally published by Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978`. The dataset is collected from [Kaggle](https://www.kaggle.com/vikrishnan/boston-house-prices/kernels). Let's get into the data and know more about it.

- **Origin**
    - The origin of the boston housing data is Natural.
- **Usage**
    - This dataset may be used for Assessment.
- **Number of Cases**
    - The dataset contains a total of 506 cases.
- **Order**
    - The order of the cases is mysterious.
- **Variables**
    There are 14 attributes in each case of the dataset. They are:
    1. CRIM - per capita crime rate by town
    2. ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS - proportion of non-retail business acres per town.
    4. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    5. NOX - nitric oxides concentration (parts per 10 million)
    6. RM - average number of rooms per dwelling
    7. AGE - proportion of owner-occupied units built prior to 1940
    8. DIS - weighted distances to five Boston employment centres
    9. RAD - index of accessibility to radial highways
    10. TAX - full-value property-tax rate per 10,000 dollars.
    11. PTRATIO - pupil-teacher ratio by town
    12. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT - % lower status of the population
    14. MEDV - Median value of owner-occupied homes in 1000's dollars

## DATA PREPROCESSING
Before performing Modeling, we will pre-process the dataset by conducting the following steps:

1. Finding Correlation between the `Predictor variables` -

A.  Correlation matrix between SalePrice with other variables <br>

![gender]("Images/Heatmap.png")

B.  SalePrice correlation matrix <br>
C.  Scatter plots between 'SalePrice' and correlated variables<br>

2. Find Missing Values and impute using `K-Means` if necessary - 
A.  Computing percent of missing Values<br>
B. Plotting the Proportion of Missing Values<br>

3. Perform Outlier Detection, to remove values that can decrease the model accuracy and lead to inappropriate predictions -
A. Univariate Analysis<br>
B. Bivariate Analysis<br>

4. Analysing the target variable `SalePrice` - 
We will check the Correlation of Target variable with Prediction variables to handle `Multi-Collinearity`. 
Also, check the skewness for 'GrLivArea' and 'TotalBsmtSF'

## DATA MODELING
After proceessing the data, we have implemented the following `Regression models` -

1. Linear Regression
2. Lasso Regression
3. Ridge Regression
4. Random Forest

## Model Evaluation
1. R-squared (R2)
2. Root Mean Square Error (RMSE)
3. Best Score
3. Cross-Validation Score
