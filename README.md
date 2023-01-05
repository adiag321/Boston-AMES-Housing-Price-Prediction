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
![Heatmap](https://user-images.githubusercontent.com/39597515/210797898-00e5218e-a623-44e8-b160-c5c091d6c89f.png)

B.  SalePrice correlation matrix <br>
![correlation_with_price](https://user-images.githubusercontent.com/39597515/210798724-f7d32b93-ac17-4b48-bc3f-72d9b8d30872.png)

2. Find Missing Values and impute using `K-Means` if necessary - 

A.  Computing percent of missing Values<br>
<img width="231" alt="Missing_values" src="https://user-images.githubusercontent.com/39597515/210797266-cffa2832-3d35-4fb3-86b2-d365bf5ee308.png">

B. Plotting the Proportion of Missing Values<br>
![Missing_values_Prop](https://user-images.githubusercontent.com/39597515/210797497-ee540036-3a08-49a7-9d40-9eea1ec5ba40.png)

3. Perform Outlier Detection, to remove values that can decrease the model accuracy and lead to inappropriate predictions - <br>
A. Univariate Analysis<br>

B. Bivariate Analysis<br>
![Bivariate_analysis](https://user-images.githubusercontent.com/39597515/210798510-36bdfc3c-ac1f-4064-aca0-e72c3f988849.png)

4. Analysing the target variable `SalePrice` - 
We will check the Correlation of Target variable with Prediction variables to handle `Multi-Collinearity`. 
Also, check the skewness for 'GrLivArea' and 'TotalBsmtSF'

## DATA MODELING
After proceessing the data, we have implemented the following `Regression models` -

1. `Linear Regression`
<img width="586" alt="Linear_regression" src="https://user-images.githubusercontent.com/39597515/210801180-1a4dab02-b195-47ef-b6bf-f92b1794f0ce.png">
2. Lasso Regression
<img width="470" alt="Lasso_regression" src="https://user-images.githubusercontent.com/39597515/210801266-ab88c903-83e2-4804-b84f-af6355f2205d.png">
3. Ridge Regression
<img width="622" alt="Ridge_regression" src="https://user-images.githubusercontent.com/39597515/210801310-389f6a73-c32f-41d6-ab95-d5e81ddd6b85.png">
4. Random Forest Regressor
<img width="914" alt="Random_Forest_Regressor" src="https://user-images.githubusercontent.com/39597515/210801054-74f1e7e7-8fb8-4125-ba98-dcc0f6b2a24a.png">

`Random Forest Regressor With different Depth Level` - 
![random_forest_diff_max_depth](https://user-images.githubusercontent.com/39597515/210801624-c22e80d8-44c9-45af-9b29-777d6d9ac469.png)

`Decision Tree Regressor` - 
![random_decision_tree_regressor](https://user-images.githubusercontent.com/39597515/210799985-366e9761-9828-4474-8339-9f483c9f9015.png)

## MODEL EVALUATION
1. R-squared (R2)
2. Root Mean Square Error (RMSE)
3. Best Score
4. Cross-Validation Score

## CONCLUSION
On the basis of our evaluation parameters calculated for each model below are the observations:
1. R-squared is a statistical measure of how close the data are to the fitted regression line. The higher the R-squared, the better the model fits the data --> `Ridge regression` (0.9285)
2. Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Lower the RMSE, the better the model fits the data --> `Ridge regression` (0.1021)
3. Higher the Best score the better the model fits the data --> `Ridge regression` (0.8857)
4. Higher Cross validation score means model performing well on the validation set, indicating that it may perform well on the unseen data(test set) --> `Ridge regression` (0.8927)
