# Regression Project MVP

## Goals

Use web scraping, exploratory data analysis, and linear regression to look at an interesting question related to statistical arbitrage.

## Initial Data Cleaning Approach and Exploratory Findings

I downloaded selected stock prices from Yahoo Finance website using an R script. I then used Python to compile the following features: 'market_cap', 'editda','day_prior_dr','comp_cor' and 'normalized stock volume'.

I relied upon normalized stock volume to investigate investment windows and ultimately chose a particular one from which to establish as my predictor (y) variable.

After cleaning the data, I examined the Scatter Matrix and learned that the variable relationships aren't generally very strong.

![Scatter_Matrix.png](Scatter_Matrix.png)

I also investigated correlations among variables which corroborated the above conclusion that factors are weakly correlated to one another.

![Feature_Correlation](Feature_Correlation.png)

After normalizing price spreads based upon stock trading volume (used Excel), I examined the mean spread ratio for all stocks relative to normalized stock volume.

![Mean_Spread_Ratio](Mean_Spread_Ratio.png)

I then established a filter (maskings) function that could be used to further refine the data set based upon particular parameters for each of 'market_cap', 'editda','day_prior_dr' and 'comp_cor'.


I ran a multiple linear regression model with `statsmodels`. The R-squared value was not high (.535) and the p-values inform me that I should probably drop some extraneous variables as I revise my model.

                          OLS Regression Results
==============================================================================
Dep. Variable:                    ror   R-squared:                       0.535
Model:                            OLS   Adj. R-squared:                  0.366
Method:                 Least Squares   F-statistic:                     3.164
Date:                Sun, 16 Apr 2017   Prob (F-statistic):             0.0585
Time:                        15:37:18   Log-Likelihood:                -63.748
No. Observations:                  16   AIC:                             137.5
Df Residuals:                      11   BIC:                             141.4
Df Model:                           4
Covariance Type:            nonrobust

                   coef    std err          t      P>|t|      [95.0% Conf. Int.]
--------------------------------------------------------------------------------
Intercept      433.8981    137.511      3.155      0.009       131.238   736.558
market_cap       1.0384      1.506      0.689      0.505        -2.277     4.354
editda         -16.5770     15.090     -1.099      0.295       -49.789    16.635
day_prior_dr  -375.4667    139.598     -2.690      0.021      -682.720   -68.213
comp_cor       -94.3460     53.054     -1.778      0.103      -211.116    22.424

Omnibus:                        2.521   Durbin-Watson:                   1.340
Prob(Omnibus):                  0.284   Jarque-Bera (JB):                1.262
Skew:                          -0.327   Prob(JB):                        0.532
Kurtosis:                       1.790   Cond. No.                         398.


Finally, I loaded the dataset into R in order to determine that the r-squared's matched (they did) and to identify the optimizated linear regression formula (using the 'step' function with direction = 'both' forward and backward).


## Intial Findings

1. **Although a relationship between explanatory and predictor variables may exist, more work needs to be done** - Althgouh the dataset reveals a possible relationship, the fact that the predictor variable (y) was based upon an optimized share volume value, raises the possibility of model overfitting.

2. **Feature Engineering** - 'day_prior_dr' was determined to be the only co-efficient which was statistically significant. Accordingly, substantial improvements could be made identifying and using more powerful features.

## Further Research and Analysis

1. Feature Engineering (mentioned above)
2. Obtain a greater number of sample observations
3. Investigate the variablitiy in investment windows and holding periods 
