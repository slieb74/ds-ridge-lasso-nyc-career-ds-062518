
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
<<<<<<< HEAD
def ridge_loss():
    #Your code here
=======
def ridge_loss(y, y_hat, coeff_weights, lam = 0.8):
    rss = np.sum((y_hat-y)**2)
    norm = np.sum(lam * coeff_weights**2)
    l2_err = rss + lam*norm
    return l2_err
>>>>>>> solution
```

## 2. Lasso (L1)
Another common normalization is called Lasso Regression and uses the $l_1$ norm.   
The ridge coefficients minimize a penalized residual sum of squares:    
    $ \sum(\hat{y}-y)^2 + \lambda\bullet |w|$

Write this loss function for performing ridge regression.


```python
<<<<<<< HEAD
def lasso_loss():
    #Your code here
=======
def lasso_loss(y, y_hat, coeff_weights, lam = 0.8):
    rss = np.sum((y_hat-y)**2)
    norm = np.sum(lam * np.abs(coeff_weights))
    l1_err = rss + lam*norm
    return l1_err
>>>>>>> solution
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
<<<<<<< HEAD
df = pd.read_excel('movie_data_detailed_with_ols.xlsx')
df.head()
X = df[['budget', 'imdbRating',
       'Metascore', 'imdbVotes']]
y = df['domgross']
=======
df = pd.read_excel('movie_data_detailed.xlsx')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>domgross</th>
      <th>title</th>
      <th>Response_Json</th>
      <th>Year</th>
      <th>imdbRating</th>
      <th>Metascore</th>
      <th>imdbVotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13000000</td>
      <td>25682380</td>
      <td>21 &amp;amp; Over</td>
      <td>0</td>
      <td>2008</td>
      <td>6.8</td>
      <td>48</td>
      <td>206513</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45658735</td>
      <td>13414714</td>
      <td>Dredd 3D</td>
      <td>0</td>
      <td>2012</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000000</td>
      <td>53107035</td>
      <td>12 Years a Slave</td>
      <td>0</td>
      <td>2013</td>
      <td>8.1</td>
      <td>96</td>
      <td>537525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61000000</td>
      <td>75612460</td>
      <td>2 Guns</td>
      <td>0</td>
      <td>2013</td>
      <td>6.7</td>
      <td>55</td>
      <td>173726</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40000000</td>
      <td>95020213</td>
      <td>42</td>
      <td>0</td>
      <td>2013</td>
      <td>7.5</td>
      <td>62</td>
      <td>74170</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df[['budget', 'imdbRating',
       'Metascore', 'imdbVotes']]
y = df['domgross']

>>>>>>> solution
def norm(col):
    minimum = col.min()
    maximum = col.max()
    return (col-maximum)/(maximum-minimum)
```


```python
<<<<<<< HEAD
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
=======
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split

#Perform test train spli
X_train , X_test, y_train, y_test = train_test_split(X, y)

ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

def rss(residual_col):
    return sum(residual_col.astype(float).map(lambda x: x**2))

print('Train Error  Ridge Model', rss(y_train - ridge_reg.predict(X_train)))
print('Test Error Ridge Model', rss(y_test - ridge_reg.predict(X_test)))
print('\n')

print('Train Error Lasso Model', rss(y_train - lasso_reg.predict(X_train)))
print('Test Error Lasso Model', rss(y_test - lasso_reg.predict(X_test)))
print('\n')

print('Train Error Unpenalized Linear Model', rss(y_train - lin_reg.predict(X_train)))
print('Test Error Unpenalized  Linear Model', rss(y_test - lin_reg.predict(X_test)))
```

    Train Error  Ridge Model 6.37432771143147e+16
    Test Error Ridge Model 1.3218369705684102e+17
    
    
    Train Error Lasso Model 6.3743249208146216e+16
    Test Error Lasso Model 1.321353862882378e+17
    
    
    Train Error Unpenalized Linear Model 6.374324920814621e+16
    Test Error Unpenalized  Linear Model 1.321353852598148e+17



```python
#For this dataset there was little difference between the models. Ridge has every so slightly better test performance.
```

# Altering Alpha

Remember that we can also change our normalization coefficient, alpha, to adjust the strenght of our normalization.
Iterate over the set **np.linspace(start=0.1, stop=2.5, num=13)** in order to find an optimal alpha.


```python
import numpy as np
min_test_error_ridge = []
min_test_error_lasso = []
optimal_ridge_alpha = 0
optimal_lasso_alpha = 0
for iter, a in enumerate(np.linspace(start=0.1, stop=2.5, num=13)):
    ridge_reg = Ridge(alpha=a)
    ridge_reg.fit(X_train, y_train)

    lasso_reg = Lasso(alpha=a)
    lasso_reg.fit(X_train, y_train)

    ridge_train_rss = rss(y_train - ridge_reg.predict(X_train))
    ridge_test_rss = rss(y_test - ridge_reg.predict(X_test))
#     print('Train Error  Ridge Model', ridge_train_rss)
#     print('Test Error Ridge Model', ridge_test_rss)
#     print('\n')
    
    lasso_train_rss = rss(y_train - lasso_reg.predict(X_train))
    lasso_test_rss = rss(y_test - lasso_reg.predict(X_test))
#     print('Train Error Lasso Model', lasso_train_rss)
#     print('Test Error Lasso Model', lasso_test_rss)
#     print('\n')
    
    if iter == 0:
        min_test_error_ridge = ridge_test_rss
        min_test_error_lasso = lasso_test_rss
        optimal_ridge_alpha = a
        optimal_lasso_alpha = a
    if min_test_error_ridge > ridge_test_rss:
        min_test_error_ridge = ridge_test_rss
        optimal_ridge_alpha = a
    if min_test_error_lasso > lasso_test_rss:
        min_test_error_lasso = lasso_test_rss
        optimal_lasso_alpha = a
print('Minimum Ridge Test RSS: {}, Best alpha: {}'.format(min_test_error_ridge, optimal_ridge_alpha))
print('Minimum Lasso Test RSS: {}, Best alpha: {}'.format(min_test_error_lasso, optimal_lasso_alpha))
```

    Minimum Ridge Test RSS: 1.321403225572795e+17, Best alpha: 0.1
    Minimum Lasso Test RSS: 1.3213538536266008e+17, Best alpha: 0.1

>>>>>>> solution
