
## Ridge and Lasso
At this point we've seen a number of criteria and algorithms for fitting regression models to data. We've seen the simple linear regression using ordinary least squares, and its more general regression of polynomial functions. We've also seen how we can arbitrarily overfit models to data using kernel methods or feature engineering. With all of that, we began to explore other tools to analyze this general problem of overfitting versus underfitting. This included train and test splits, bias and variance, and cross validation.

Now we're going to take a look at another way to tune our models. These methods all modify our mean squared error function that we were optimizing against. The modifications will add a penalty for large coefficient weights in our resulting model. If we think back to our case of feature engineering, we can see how this penalty will help combat our ability to create more accurate models by simply adding additional features.

In general, all of these penalties are known as $L^p norms$.

## $L^p$ norm of x
In order to help account for underfitting and overfitting, we often use what are called $L^p$ norms.   
The **$L^p$ norm of x** is defined as:  

### $||x||_p  =  \big(\sum_{i} x_i^p\big)^\frac{1}{p}$

## 1. Ridge (L2)
One common normalization is called Ridge Regression and uses the $l_2$ norm (also known as the Euclidean norm) as defined above.   
The ridge coefficients minimize a penalized residual sum of squares:    
    $ \sum(\hat{y}-y)^2 + \lambda\bullet w^2$

Write this loss function for performing ridge regression.


```python
import numpy as np
```


```python
def ridge_loss():
    #Your code here
```

## 2. Lasso (L1)
Another common normalization is called Lasso Regression and uses the $l_1$ norm.   
The ridge coefficients minimize a penalized residual sum of squares:    
    $ \sum(\hat{y}-y)^2 + \lambda\bullet |w|$

Write this loss function for performing ridge regression.


```python
def lasso_loss():
    #Your code here
```

## 3. Run + Compare your Results
Run a ridge lasso and unpenalized regressions on the dataset below.
While we have practice writing the precursors to a full ridge regression, we'll import the package for now.
Then, answer the following questions:
* Which model do you think created better results overall? 
* Comment on the differences between the coefficients of the resulting models


```python
import pandas as pd
```


```python
df = pd.read_excel('movie_data_detailed_with_ols.xlsx')
df.head()
X = df[['budget', 'imdbRating',
       'Metascore', 'imdbVotes']]
y = df['domgross']
def norm(col):
    minimum = col.min()
    maximum = col.max()
    return (col-maximum)/(maximum-minimum)
```


```python
from sklearn.model_selection import train_test_split
```


```python
from sklearn.linear_model import Lasso, Ridge, LinearRegression
```


```python
#Fit the Ridge Model
```


```python
#Fit the Lasso Model
```


```python
#Fit the Unpenalized Model
```


```python
#Calculate the test and train error for all 3 models.
```
