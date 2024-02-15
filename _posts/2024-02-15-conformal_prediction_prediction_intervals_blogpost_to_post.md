---
layout: post
title: Conformal Quantile Regression
subtitle: Introduction and implementation using scikit-learn
tags: [machine learning]
comments: false
author: Vincent Wauters
---


# Intro
Suppose you have built a fantastic machine learning model for predicting the selling price of a given house. Between the following two statements, what will make the biggest impression?

_'I predict that this house has a value of 450.000 euro'_

_or_

_'I predict that this house has a value between 435.000 euro and 465.000 euro with 90% certainty_'

While the boldness of the first statement might impress some people - certainly in the area of real estate - the later statement does convey information about the magnitude of uncertainty, which is incredibly useful. This is uncertainty quantification (UQ), for regression cases this comes down to constructing _prediction intervals_ (PI) and for classification to _prediction sets_.

Up until recently, the most common ways of uncertainty quantification were based on unrealistic assumptions (such as normality) amongst others and led to unsatisfactory results. Luckily, there is a 'new(er)' kid on the block called _conformal prediction_ or _conformal inference_, which is a very powerful and model-agnostic method to construct reliable prediction intervals or predictions sets without any distributional assumptions. 

A specific type of conformal prediction leverages the power of (conditional) quantile regression and provides a rather elegant way of constructing prediction intervals for regression cases that have nice properties, this will be the focus of this post.

# Conformal Quantile Regression: Concepts

## Background

Before diving straight into the specific type of conformal prediction of focus, let's quickly provide some more context about conformal prediction. Research in this field has been going on for some decades, mainly focused around the work of Vladimir Vovk and his colleagues.

Types of conformal prediction are often divided into 3 categories:
1. Full conformal prediction (the original implementation)
2. Split-conformal prediction
3. Cross-conformal prediction

Full conformal prediction, which is the original way of conducting conformal prediction following Vovk's research, requires vof refitting a very large amount of models, hence is computationally heavy. The latter two make use of data splitting that drive down the computation load substantially. 

## Conditional Quantile Regression and the Pinball Loss

For the case of regression with continuous outcomes, __(conditional) quantile regression__ is well known, but not particularly popular. While most regression models estimate the conditional mean, quantile regression aims to estimate a certain conditional quantile $q_{\tau}$ with $\tau \in [0, 1]$. The most common example would be to estimate the conditional $\tau=0.5$ quantile $q_{0.5}$, which is the conditional median. And yes indeed, one could estimate two conditional quantile models, one for lower bound and one for an upper bound to try to achieve valid prediction intervals. More on this a little later!

How can one estimate a conditional quantile instead of the (usual) conditional mean, you ask? Quantile regression achieves this by using a specific family of loss functions. Just like estimating the conditional mean is done by minimizing the mean squared error MSE loss, estimating conditional quantiles is achieved by minimizing the family of __pinball losses__ or _hockey stick losses_. Take the loss of the target and the estimated quantile using the features $\mathbf{x}$ as $q(\mathbf{x})$, for a conditional quantile $q_{\tau}$ with $\tau \in [0, 1]$

$$\rho_{\tau}(y, \hat{q}(\mathbf{x})) :=
\begin{cases}
    \tau(y-\hat{q}(\mathbf{x})), & \text{if } y-\hat{q}(\mathbf{x}) > 0\\
    (1-\tau)(y - \hat{q}(\mathbf{x})), & \text{otherwise.}
\end{cases}
$$

With some simplification by looking at the difference between target and predicted quantile as a kind of residual $\hat{\varepsilon} = y - \hat{q}(\mathbf{x})$ and taking the average, one can simplify the loss as:

__TODO IS THIS EQUIVALENT__

$$ \rho_{\tau}(y, \hat{q}(\mathbf{x})) := max(\tau \hat{\varepsilon}; (1 - \tau) \hat{\varepsilon})$$

And for all samples, taking the average as is usual:

$$\frac{1}{n} \sum_{i=1}^{n} max(\tau \hat{\varepsilon}_i; (1 - \tau) \hat{\varepsilon}_i)$$

In simple terms: this loss allows for incurring different losses for equal underprediction or overprediction magnitudes which can be seen by the differing angles to the left and the right. Also notice that if the prediction is equal to the target, the loss is 0, which is obviously a property that is desirable for a loss function.


![pinball_loss.png](attachment:pinball_loss.png)

To fit quantile regression models in practice, Gradient Boosting Machines (GBM) are a natural candidate. Instead of using a squared error loss, as is usual, one just uses the appropriate pinball loss.

As stated earlier, a potentially simple scheme for estimating prediction intervals arises almost naturally here: by estimating two conditional quantiles, a lower bound, $q_L$ ,and an upper bound $q_U$. 

The quantile levels $L, U \in [0, 1]$ to estimate are then chosen to achieve a certain _coverage_ $(1 - \alpha)$ and _mis-coverage_ $\alpha$ levels. Some examples:

* __A 90% symmetric prediction interval:__
    * mis-coverage is 10%:  $\alpha = 0.1$. 
    * coverage is 90%: $1 - \alpha = 0.9$
    * estimate conditional quantiles: $q_{0.05}(\mathbf{x})$ and $q_{0.95}(\mathbf{x})$, because miscoverage is distributed equally over lower and upper bounds
* __A 80% symmetric prediction interval:__
    * mis-coverage is 20%: , $\alpha = 0.2$. 
    * coverage is 80%: $1 - \alpha = 0.8$
    * estimate conditional quantiles for quantiles: $q_{0.1}(\mathbf{x})$ and $q_{0.9}(\mathbf{x})$
* __A 90% asymmetric prediction interval:__
    * mis-coverage is 10%:, $\alpha = 0.1$.
    * coverage is 90%:  $1 - \alpha = 0.9$
    * dsitribute mis-coverage for the lower and upper bound arises as desired, some examples:
        * 2.5% mis-coverage for the lower bound, 7.5% for the upper bound: estimate conditional quantile models for $q_{0.025}(\mathbf{x})$ and $q_{0.925}(\mathbf{x})$
        * 8% mis-coverage for the lower bound, 2% for the upper bound: estimate conditional quantile models for $q_{0.08}(\mathbf{x})$ and $q_{0.98}(\mathbf{x})$.


We could just stop here, because, estimating models by the appropriate pinball losses already guarantees convergence (statistcal consistency) to the population conditional quantiles, ... asymptotically $(n \to +\infty)$. In finite samples, it has been shown that the actual coverage is far off from the required (nominal) coverage. 

However, Conformalized conditional Quantile Regression (CQR), will take the main idea from quantile regression and augment it. This is what we will see next.

## (Non)-conformity Score

One of the most pivotal concepts in conformal prediction is the __(non)-conformity score $s_i$__. This score serves to encode (measure) the disagreement between predictions and targets. Many options exist on what kind of functional form of non-conformity score to choose. It can be as something simple as the absolute residual. However, since we are dealing with two conditional quantile regression models here, we need to apply some additional trickery

Suppose you have chosen two $\tau$-values, a lower and upper value $(L, U)$ (e.g. $L=0.05$ and $U=0.95$) for your two predicted conditional quantiles $\hat{q}_L(\mathbf{x}), \hat{q}_U(\mathbf{x})$ to construct a prediction interval. 

One elegant option that works well for this case is the following non-conformity score which can be calculated for each observation $i=1, ..., n$

$$s_i(y_i, \hat{q}(\mathbf{x}_i)) = \max\{\hat{q}_L(\mathbf{x}_i) - y ~;~ y - \hat{q}_U(\mathbf{x}_i)\}$$

In essence: for observations where $y_i$ falls within the prediction interval range, both values are negative and the distance to the closest boundary is taken as non-conformity score. For observations where the target falls outside the prediction interval: the distance to the closest boundary is taken.

At this point we have $n$ non-conformity scores $s_i$, or a vector $\mathbf{s}$. What to do with these? The idea is simple: __from this vector of $s_i$, take the (1 - $\alpha$)'th quantile and use this as a correction factor to adjust the predicted lower and upper conditional quantiles.__

$Quantile(s_1, ..., s_n; (1-\alpha))$

A slight modification is done to get more attractive finite sample properties.
 
$s_{adj} = Quantile(s_1, ..., s_n; (\frac{ceiling[(n + 1)(1-\alpha)]}{n}))$

Note that for very large $n$, this adjust value asymptotically will converge to the unadjusted (ordinary) sample quantile. Related is the fact that quantile regression can handle heteroskedastic noise, i.e. for certain regions, the uncertainty might be higher than for others (often this makes sense). 

Then, one simple thing needs to be done: adjust the resulting prediction interval using this calculated $s_{adj}$:


$$C(\mathbf{x}_i) = \left[ \hat{q}_{L}(\mathbf{x}_i) - s_{adj} ~ ; ~ \hat{q}_{U}(\mathbf{x}_i) + s_{adj} \right]$$


## Data Splitting Strategies: Split-Conformal Learning and Cross-Conformal Learning

Originally, full conformal learning, as was hinted before in the _Background_ section, led to a heavy computational load due to refitting $n$ models. Very similarly (and not by coincidence) to how data is often split to measure generalization performance, one can also choose to split data to conformalize or calibrate a models' predictions. Two flavours exist:

1. Split-Conformal learning
2. Cross-Conformal learning

Split conformal prediction divides all available data into two disjoint sets:
* $D_1$: data used for training, the proper training set to train a machine learning model on
* $D_2$: data used for conformalizing or calibrating the trained machine learning model to achieve proper coverage

While cross-conformal learning will divide the dataset into $K$ disjoint groups, and similarly to cross-validation, will have one of the groups play the role of the calibration set, while the others together form the proper training set and repeat this $K$ times until each datapoint has been part of the calibration set exactly once and $K-1$ times member of the proper training set. 

The reason why we apply data splitting in the first place is very similar to why measuring loss on the training set might be misleading: we don't want optimistically bias the (non-)conformity scores $s_i$ and hence the resulting adjustment factor. Such a bias would lead to worse results on unseen data, i.e. having incorrect coverages on unseen data.


# Conformal Quantile Regression in Python

Let's now see how we can use conformal quantile regression in Python. First we will load libraries and the _Ames Housing_ data and perform some basic preprocessing on it.

## Loading Libraries


```python
# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pprint

# For reproduceability, showing which sklearn version is used. Use a modern version ideally >= 1.2
import sklearn; print(f"Scikit-learn version used: {sklearn.__version__}")

from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import (train_test_split)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import fetch_openml
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_pinball_loss
```

    Scikit-learn version used: 1.3.0
    


```python
# Setting some parameters
plt.style.use("ggplot")  # Cleaner plotting
```


```python
# Setting seed
seed = 42
```

## Loading Data


```python
# Loading Ames housing data
housing = fetch_openml(name="house_prices", 
                       as_frame=True, 
                       parser='auto')

# Getting features (X) and target (y)
X = housing['data']
y = housing['target']

print(f"X (feature matrix) shape: {X.shape}")
print(f"y (target vector) shape: {y.shape}")
```

    X (feature matrix) shape: (1460, 80)
    y (target vector) shape: (1460,)
    

## Simple Preprocessing


```python
# Quick look, appearantly, extremely skewed:
plt.hist(y, bins=50, color='salmon', alpha=0.8);
plt.title('Histogram of y  (house prices)')
plt.show();

plt.hist(np.log(y), bins=50, color='salmon', alpha=0.8);
plt.title('Histogram of log(y) ($log(house prices)$)')
plt.show();

# Maybe cut off the top? -- Check ECDF - Easier than Hist!...

# Cutting off the top 10%?
q_cutoff = 0.9
y_cutoff = np.quantile(a=y, q=q_cutoff)
plt.hist(y[y <= y_cutoff], bins=50, color='salmon', alpha=0.8);
plt.title('Histogram of y, cutoff at 90th quantile')
plt.show()
```


    
![png](output_20_0.png)
    



    
![png](output_20_1.png)
    



    
![png](output_20_2.png)
    



```python
# Decision: let's cutoff at 90% to get some more nicely behaving distribution
remove_y_outliers = True
if remove_y_outliers:
    to_keep_idx = np.where(housing['target'] <= np.quantile(housing['target'], 0.9))
    y = housing['target'].loc[to_keep_idx]
    X = housing['data'].loc[to_keep_idx]
```


```python
# Selecting some (mainly numeric) columns only
feat_cols = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
             'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr',
             'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF']
X = X[feat_cols].copy()
```

Let's split the data into 3 parts for now:
* $D_1$: the _proper training set_ used for training the model. This part will be also divided into parts using cross-validation for tuning hyperparameters. (60% of the observations)
* $D_2$: the _calibration set_ used to conformalize the prediction intervals. (50% of 40% = 20% of observations)
* $D_3$: a completely unseen test set that can be used to see if the coverages hold in unseen data. (50% of 40% = 20% of observations)


```python
# Splitting in train-calibration-test set
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=seed)
X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, shuffle=True, random_state=seed)

print(f'Sample sizes for train set: {len(y_train)} \n Sample size for calibration set: {len(y_cal)} \n Sample size for test set: {len(y_test)}')
```

    Sample sizes for train set: 789 
     Sample size for calibration set: 263 
     Sample size for test set: 263
    

## Training and Tuning Conditional Quantile Regression GBMs

Let's use a very general approach for training and tuning GBM models, with the big differences being:
* Choosing a coverage level $1 - \alpha$, and choosing how to allocate the mis-coverage over the lower and upper boundaries. We will choose a symmetric 90% prediction interval, hence: $q_L = q_{0.05}$ and $q_U = q_{0.95}$.
* Training and tuning of the two conditional quantile models instead of a single (conditional mean) model.
* Use of the appropriate pinball losses, both for training and tuning. 

Except this, the procedure we will follow is quite common:

1. Defining a hyperparameter grid. We'll take some basic hyperparameters and define reasonable values for them. We will use the same grid for both conditional quantiles we estimate.
2. Define scoring metrics for each model using `make_scorer()`, for conditional mean models you'd usually just use default scoring.
3. Instantiate a GBM for each quantile to estimate. We will use the newer `HistGradientBoostingRegressor` that takes inspiration from XGBoost and LightGBM implementations, in recent versions this supports estimation of quantiles by defining `loss=quantile` and supplying the requested $\tau$ to the `quantile` argument.
4. Define a data splitting strategy, for this we will use a repeated K-fold cross-validation setup using the built-in `RepeatedKFold` class. We'll use 2-times repeated 4-fold cross-validation.
5. Search the hyperparameter grid by sampling using random search (`RandomizedSearchCV`), we will only search 10 values, for simplicity sake.
6. Obtain the optimal parameters from the 


```python
# Generating hyperparameter grid
param_grid = {'learning_rate': [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
              'max_iter': [50, 100, 150, 250],  # amount of trees
              'max_depth': [2, 3, 5, 7],
              'min_samples_leaf': [3, 5, 10, 20, 30]}        

# Defining mis-coverage level alpha
alpha = 0.10

# Defining tau to predict the quantile q_tau for
taus = [0.05, 0.95]  # contains (L, U) levels

# Setting some other CV params
n_hyperparams = 20
n_folds = 4
n_repeats = 2
seed = 42

# Create the appropriate pinball loss functions
pinball_losses = [make_scorer(mean_pinball_loss, alpha=tau, greater_is_better=False) for tau in taus]
pinball_losses = dict(zip(taus, pinball_losses))  # Dictionary with the alphas as keys and losses as values

# Cross-validation with chosen search strategy for hyperparameter values
models = dict.fromkeys(taus, None)  # Dictionary with keys being the tau values to save models in
tuning_objects = dict.fromkeys(taus, None)  # Same, but to save tuned model in
for tau in taus:
    print(f'Tuning hyperparameters for conditional quantile regression model: q_{tau}')
    models[tau] = HistGradientBoostingRegressor(loss='quantile', quantile=tau, random_state=seed) 
    # From sklearn v1.1 onwards you can use faster HistGradientBoostingRegressor instead of GradientBoostingRegressor
    
    # Repeated K-fold cross-validation data resampling
    cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    
    # Tuning using random grid search
    tuning_objects[tau] = RandomizedSearchCV(estimator=models[tau],
                                             cv=cv,
                                             param_distributions=param_grid,
                                             n_iter=n_hyperparams,
                                             scoring=pinball_losses[tau],
                                             random_state=seed,
                                             verbose=1)
    tuning_objects[tau].fit(X=X_train, y=y_train)
    print(f'Optimal hyperparameters {tuning_objects[tau].best_params_}')
    print('*' * 125)
```

    Tuning hyperparameters for conditional quantile regression model: q_0.05
    Fitting 8 folds for each of 20 candidates, totalling 160 fits
    

    C:\Users\Vincent Wauters\anaconda3\envs\flames_ml_2023\Lib\site-packages\sklearn\model_selection\_search.py:987: RuntimeWarning: overflow encountered in square
      (array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights
    

    Optimal hyperparameters {'min_samples_leaf': 3, 'max_iter': 150, 'max_depth': 2, 'learning_rate': 0.1}
    *****************************************************************************************************************************
    Tuning hyperparameters for conditional quantile regression model: q_0.95
    Fitting 8 folds for each of 20 candidates, totalling 160 fits
    

    C:\Users\Vincent Wauters\anaconda3\envs\flames_ml_2023\Lib\site-packages\sklearn\model_selection\_search.py:987: RuntimeWarning: overflow encountered in square
      (array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights
    

    Optimal hyperparameters {'min_samples_leaf': 30, 'max_iter': 150, 'max_depth': 2, 'learning_rate': 0.2}
    *****************************************************************************************************************************
    

Let's now use the optimally (re)-trained models for the lower and upper quantiles and predict instances on the (unseen) calibration data.


```python
# Gather predictions on calibration data, putting them in a N x 2 NumPy array
lower_preds_cal = tuning_objects[taus[0]].predict(X=X_cal)
upper_preds_cal = tuning_objects[taus[1]].predict(X=X_cal)
preds_cal = np.vstack([lower_preds_cal, upper_preds_cal]).T  # Transpose to get N x 2 instead of 2 x N

lower_preds_test = tuning_objects[taus[0]].predict(X=X_test)
upper_preds_test = tuning_objects[taus[1]].predict(X=X_test)
preds_test = np.vstack([lower_preds_test, upper_preds_test]).T

# Make them into Pandas DataFrames with correct indices (predict returns NumPy arrays...)
preds_cal_pd = pd.DataFrame(preds_cal, index=X_cal.index, columns=['lower', 'upper'])
preds_test_pd = pd.DataFrame(preds_test, index=X_test.index, columns=['lower', 'upper'])
```

## Conformalizing the Quantile Regression Models

We take the predictions on the calibration dataset ($D_2$) and translate the theory into code. For this we will make a small function `calculate_conformal_correction`.


```python
def calculate_conformal_correction(preds_cal, target_cal, alpha):
    """
    Calculates conformity scores and calculates the modified quantile to correct the raw prediction intervals
    provided to achieve correct requested coverage and mis-coverage (alpha). Ideally this is performed on a 
    separate calibration dataset (hence pred_cal, target_cal).
    """
    # Calculate the 'deviations' from lower and upper predicted quantiles
    preds_lower = preds_cal[:, 0]  # First col is lower bound
    preds_upper = preds_cal[:, 1]  # Second col is upper bound
    dev = np.vstack([preds_lower - target_cal, target_cal - preds_upper])
    nonconf_scores = np.max(dev, axis=0)  # max of each 2-pair of deviations: i=1, ..., N
    
    # Take the 'modified' quantile of these non-conformity scores 
    n = len(target_cal)
    alpha_compl_mod = np.ceil((n + 1) * (1 - alpha)) / n
    correction = np.quantile(a=nonconf_scores, q=alpha_compl_mod)
    
    return correction
```


```python
# Use function to calculate the correction factor
conformal_correction = calculate_conformal_correction(preds_cal=preds_cal,
                                                      target_cal=y_cal,
                                                      alpha=0.1)
print(f"The conformal correction calculated on the calibration data is: {conformal_correction: .2f} (USD)")

# Compare the correction magnitude to average of target in calibration set to get relative idea
print(f"The conformal correction relative to the average house price: {conformal_correction / y_cal.mean() * 100:.2f}%")
```

    The conformal correction calculated on the calibration data is:  6039.98 (USD)
    The conformal correction relative to the average house price: 3.82%
    

The calculated correction factor to adjust both lower and upper (raw) prediction bounds is (+) 6039 (USD). This is approximately 3.82% of the average house, hence nothing dramatic. This seems very reasonable compared to the magnitude of house prices. 

We can now progress by just applying the adjustment, since it's positive it means that we will enlarge the prediction interval, hence moving the lower bound down and the upper bound up. 


```python
# Conformalize the calibration and test predictions using the calculated correction
lower_preds_cal_conformal = lower_preds_cal - conformal_correction
upper_preds_cal_conformal = upper_preds_cal + conformal_correction

lower_preds_test_conformal = lower_preds_test - conformal_correction
upper_preds_test_conformal = upper_preds_test + conformal_correction
```

## Conforming Coverage

Let's now check, on the unseen test data set, if the requested coverage is achieved. We make a very simple function to 


```python
def calculate_coverage(preds, target):
    """
    Calculates if the target falls within the lower and upper bounds to calculate actual coverage.
    """
    # Count observations where target is within bounds (1 if yes, 0 if not)
    target_in_bounds_indicator = np.where((preds[:, 0] <= target) & (preds[:, 1] > target), 1, 0)
    
    # Return average (=proportion with the 0/1 from above)
    return np.mean(target_in_bounds_indicator)
```


```python
# Use function for the raw predictions on test set and the conformalized ones
preds_test_conformal = np.vstack([lower_preds_test_conformal, upper_preds_test_conformal]).T

coverage_test_raw = calculate_coverage(preds=preds_test, target=y_test)
coverage_test_conformal = calculate_coverage(preds=preds_test_conformal, target=y_test)

print(f"** Requested coverage levels: {1 - alpha: .2f} **")
print(f"Actual coverage on test set without conformal correction: {coverage_test_raw: .2f}")
print(f"Actual coverage on test set with conformal correction: {coverage_test_conformal: .2f}")
```

    ** Requested coverage levels:  0.90 **
    Actual coverage on test set without conformal correction:  0.78
    Actual coverage on test set with conformal correction:  0.86
    

We see that, using the correction calculated on the separate calibration set, also leads to getting nearer to requested coverage on unseen data. We can also performan a sanity check that on the calibration set itself, the correction achieves (near) perfect coverage.


```python
# Sanity check: coverage on the calibration set itself should be near perfect
preds_cal_conformal = np.vstack([lower_preds_cal_conformal, upper_preds_cal_conformal]).T

coverage_cal_raw = calculate_coverage(preds=preds_cal, target=y_cal)
coverage_cal_conformal = calculate_coverage(preds=preds_cal_conformal, target=y_cal)

print(f"** Requested coverage levels: {1 - alpha: .2f} **")
print(f"Actual coverage on calibration set without conformal correction: {coverage_cal_raw: .2f}")
print(f"Actual coverage on calibration set without conformal correction: {coverage_cal_conformal: .2f}")
```

    ** Requested coverage levels:  0.90 **
    Actual coverage on calibration set without conformal correction:  0.80
    Actual coverage on calibration set without conformal correction:  0.90
    

Yes it does work as expected: going from 80% coverage without conformalizing to 90% as was requested!


```python
def create_conformal_summary_df(preds_cal, preds_test, target_cal, target_test, alpha, alpha_lower=None):
    """
    Create both symmetric and asymmetric conformalized preds and make an overview DF including coverage.
    
    """
    
    preds_c_symmetric = conformalize_preds(preds_cal=preds_cal, 
                                           preds_test=preds_test, 
                                           target_cal=target_cal,
                                           alpha=alpha,
                                           mode='quantile')
    preds_c_asymmetric = conformalize_preds(preds_cal=preds_cal, 
                                            preds_test=preds_test,
                                            target_cal=target_cal,
                                            alpha=alpha,
                                            alpha_lower=alpha_lower,
                                            mode='quantile_asymmetric')
    
    # Making overview DataFrame
    df = pd.DataFrame({'lower_pred_test_q': lower_preds_test, 
                       'upper_pred_test_q': upper_preds_test})
    
    df['y_test'] = target_test
    df['lower_pred_test_symm'] = preds_c_symmetric[:, 0]
    df['upper_pred_test_symm'] = preds_c_symmetric[:, 1]
    df['lower_pred_test_asymm'] = preds_c_asymmetric[:, 0]
    df['upper_pred_test_asymm'] = preds_c_asymmetric[:, 1]
    df['coverage_q'] = np.where((df['lower_pred_test_q'] <= df['y_test']) & (df['y_test'] <= df['upper_pred_test_q']), 1, 0) # Correct?
    df['coverage_symm'] = np.where((df['lower_pred_test_symm'] <= df['y_test']) & (df['y_test'] <= df['upper_pred_test_symm']), 1, 0)
    df['coverage_asymm'] = np.where((df['lower_pred_test_asymm'] <= df['y_test']) & (df['y_test'] <= df['upper_pred_test_asymm']), 1, 0)
    df['width_q'] = np.abs(df['upper_pred_test_q'] - df['lower_pred_test_q'])
    df['width_symm'] = np.abs(df['upper_pred_test_symm'] - df['lower_pred_test_symm'])
    df['width_asymm'] = np.abs(df['upper_pred_test_asymm'] - df['lower_pred_test_asymm'])

    # Printing some diagnostics
    print(f"Coverage for original quantile predictions is: {df['coverage_q'].mean():.3f}")
    print(f"Coverage for symmetric conformalized predictions is: {df['coverage_symm'].mean():.3f}")
    print(f"Coverage for asymmetric conformalized predictions is: {df['coverage_asymm'].mean():.3f}")

    return df
```


```python
df_summary = create_conformal_summary_df(preds_cal=[lower_preds_cal, upper_preds_cal],
                                         preds_test=[lower_preds_test, upper_preds_test],
                                         target_cal=y_cal, 
                                         target_test=y_test,
                                         alpha=0.1,
                                         alpha_lower=0.05)
print('Showing 5 top rows of the predictions with various bounds...')
display(df_summary.head())
```

    q_hat 714290604.7937503
    delta q_hat 1428581209.5875006
    q_hat [513453828.5726157, 1207842014.0009553]
    delta q_hat 1721295842.573571
    Coverage for original quantile predictions is: 0.801
    Coverage for symmetric conformalized predictions is: 0.899
    Coverage for asymmetric conformalized predictions is: 0.931
    Showing 5 top rows of the predictions with various bounds...
    


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
      <th>lower_pred_test_q</th>
      <th>upper_pred_test_q</th>
      <th>y_test</th>
      <th>lower_pred_test_symm</th>
      <th>upper_pred_test_symm</th>
      <th>lower_pred_test_asymm</th>
      <th>upper_pred_test_asymm</th>
      <th>coverage_q</th>
      <th>coverage_symm</th>
      <th>coverage_asymm</th>
      <th>width_q</th>
      <th>width_symm</th>
      <th>width_asymm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1366</th>
      <td>1.695514e+10</td>
      <td>2.305570e+10</td>
      <td>19300000000</td>
      <td>1.624084e+10</td>
      <td>2.376999e+10</td>
      <td>1.644168e+10</td>
      <td>2.426354e+10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6.100560e+09</td>
      <td>7.529142e+09</td>
      <td>7.821856e+09</td>
    </tr>
    <tr>
      <th>449</th>
      <td>9.054818e+09</td>
      <td>1.308768e+10</td>
      <td>12000000000</td>
      <td>8.340528e+09</td>
      <td>1.380197e+10</td>
      <td>8.541365e+09</td>
      <td>1.429552e+10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4.032862e+09</td>
      <td>5.461444e+09</td>
      <td>5.754158e+09</td>
    </tr>
    <tr>
      <th>266</th>
      <td>1.519322e+10</td>
      <td>1.915203e+10</td>
      <td>18500000000</td>
      <td>1.447893e+10</td>
      <td>1.986632e+10</td>
      <td>1.467976e+10</td>
      <td>2.035987e+10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3.958808e+09</td>
      <td>5.387390e+09</td>
      <td>5.680104e+09</td>
    </tr>
    <tr>
      <th>518</th>
      <td>1.557716e+10</td>
      <td>1.996983e+10</td>
      <td>21100000000</td>
      <td>1.486287e+10</td>
      <td>2.068412e+10</td>
      <td>1.506371e+10</td>
      <td>2.117767e+10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4.392673e+09</td>
      <td>5.821254e+09</td>
      <td>6.113969e+09</td>
    </tr>
    <tr>
      <th>277</th>
      <td>8.360104e+09</td>
      <td>1.597815e+10</td>
      <td>14100000000</td>
      <td>7.645813e+09</td>
      <td>1.669244e+10</td>
      <td>7.846650e+09</td>
      <td>1.718600e+10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7.618049e+09</td>
      <td>9.046631e+09</td>
      <td>9.339345e+09</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_summary = create_conformal_summary_df(preds_cal=[lower_preds_cal, upper_preds_cal],
                                         preds_test=[lower_preds_test, upper_preds_test],
                                         target_cal=y_cal, 
                                         target_test=y_test,
                                         alpha=0.1)
df_summary.head()
```

    q_hat 714290604.7937503
    delta q_hat 1428581209.5875006
    Warning asymmetric mode but alpha_lower not given! Dividing alpha separately over bounds...
    q_hat [513453828.5726157, 1207842014.0009553]
    delta q_hat 1721295842.573571
    Coverage for original quantile predictions is: 0.801
    Coverage for symmetric conformalized predictions is: 0.899
    Coverage for asymmetric conformalized predictions is: 0.931
    




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
      <th>lower_pred_test_q</th>
      <th>upper_pred_test_q</th>
      <th>y_test</th>
      <th>lower_pred_test_symm</th>
      <th>upper_pred_test_symm</th>
      <th>lower_pred_test_asymm</th>
      <th>upper_pred_test_asymm</th>
      <th>coverage_q</th>
      <th>coverage_symm</th>
      <th>coverage_asymm</th>
      <th>width_q</th>
      <th>width_symm</th>
      <th>width_asymm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1366</th>
      <td>1.695514e+10</td>
      <td>2.305570e+10</td>
      <td>19300000000</td>
      <td>1.624084e+10</td>
      <td>2.376999e+10</td>
      <td>1.644168e+10</td>
      <td>2.426354e+10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6.100560e+09</td>
      <td>7.529142e+09</td>
      <td>7.821856e+09</td>
    </tr>
    <tr>
      <th>449</th>
      <td>9.054818e+09</td>
      <td>1.308768e+10</td>
      <td>12000000000</td>
      <td>8.340528e+09</td>
      <td>1.380197e+10</td>
      <td>8.541365e+09</td>
      <td>1.429552e+10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4.032862e+09</td>
      <td>5.461444e+09</td>
      <td>5.754158e+09</td>
    </tr>
    <tr>
      <th>266</th>
      <td>1.519322e+10</td>
      <td>1.915203e+10</td>
      <td>18500000000</td>
      <td>1.447893e+10</td>
      <td>1.986632e+10</td>
      <td>1.467976e+10</td>
      <td>2.035987e+10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3.958808e+09</td>
      <td>5.387390e+09</td>
      <td>5.680104e+09</td>
    </tr>
    <tr>
      <th>518</th>
      <td>1.557716e+10</td>
      <td>1.996983e+10</td>
      <td>21100000000</td>
      <td>1.486287e+10</td>
      <td>2.068412e+10</td>
      <td>1.506371e+10</td>
      <td>2.117767e+10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4.392673e+09</td>
      <td>5.821254e+09</td>
      <td>6.113969e+09</td>
    </tr>
    <tr>
      <th>277</th>
      <td>8.360104e+09</td>
      <td>1.597815e+10</td>
      <td>14100000000</td>
      <td>7.645813e+09</td>
      <td>1.669244e+10</td>
      <td>7.846650e+09</td>
      <td>1.718600e+10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7.618049e+09</td>
      <td>9.046631e+09</td>
      <td>9.339345e+09</td>
    </tr>
  </tbody>
</table>
</div>



## Visualization

Let's try to get an intuitive visualization of what is happening by taking a selection of observations (e.g. 100) and plotting them with their associated raw prediction intervals straight from the conditional quantile regression and the conformalized equivalents.


```python
# Let's put everything in a DF first
df_summary = pd.DataFrame(data={
    'y_test': y_test,
    'lower_preds_test': lower_preds_test,
    'upper_preds_test': upper_preds_test,
    'lower_preds_test_conformal': lower_preds_test_conformal,
    'upper_preds_test_conformal': upper_preds_test_conformal
}, index=y_test.index)

# Check results
df_summary.head(5)
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
      <th>y_test</th>
      <th>lower_preds_test</th>
      <th>upper_preds_test</th>
      <th>lower_preds_test_conformal</th>
      <th>upper_preds_test_conformal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>682</th>
      <td>173000</td>
      <td>157034.167892</td>
      <td>195471.258204</td>
      <td>150994.186533</td>
      <td>201511.239563</td>
    </tr>
    <tr>
      <th>366</th>
      <td>159000</td>
      <td>139621.046374</td>
      <td>191948.243580</td>
      <td>133581.065015</td>
      <td>197988.224939</td>
    </tr>
    <tr>
      <th>883</th>
      <td>118500</td>
      <td>99404.064449</td>
      <td>195118.918333</td>
      <td>93364.083090</td>
      <td>201158.899692</td>
    </tr>
    <tr>
      <th>413</th>
      <td>115000</td>
      <td>81565.288830</td>
      <td>131462.759300</td>
      <td>75525.307471</td>
      <td>137502.740660</td>
    </tr>
    <tr>
      <th>68</th>
      <td>80000</td>
      <td>60689.654351</td>
      <td>112367.522110</td>
      <td>54649.672992</td>
      <td>118407.503469</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting sample of some observations
# First, sample some observations
n_sample = 10
preds_test_sample = df_summary.sample(n_sample, random_state=seed).reset_index()  # reset_index, it makes plot cleaner

# Plotting
plt.figure(figsize=(20, 10))

# Plotting the PIs with vertical lines and whiskers
# For raw prediction intervals
plt.errorbar(x=preds_test_sample.index - 0.01, # small adjustment to not have the 2 bars overlap perfectly
             y=(preds_test_sample['lower_preds_test'] + preds_test_sample['upper_preds_test']) / 2,  # "midpoint of the PI"
             yerr=[(preds_test_sample['upper_preds_test'] - preds_test_sample['lower_preds_test']) / 2],  # half the width around the point
             fmt=' ', color='lightblue', ecolor='cadetblue', elinewidth=3, capsize=5, capthick=2, alpha=0.7,
             label='Raw Prediction Interval')

# For conformal prediction intervals
plt.errorbar(x=preds_test_sample.index + 0.01, # small adjustment to not have the 2 bars overlap perfectly
             y=(preds_test_sample['lower_preds_test_conformal'] + preds_test_sample['upper_preds_test_conformal']) / 2,  # "midpoint of the PI"
             yerr=[(preds_test_sample['upper_preds_test_conformal'] - preds_test_sample['lower_preds_test_conformal']) / 2],  # half the width around the point
             fmt=' ', color='lightcoral', ecolor='salmon', elinewidth=3, capsize=5, capthick=2, alpha=0.7,
             label='Conformal Prediction Interval')

# Plot true target values
plt.plot(preds_test_sample.index, preds_test_sample['y_test'], 'o', markersize=10, color='darkslategray', label='True target value')
plt.title(f"Sample of {n_sample} houses and their raw PI and conformalized PI", fontsize=15)
plt.xlabel('House', fontsize=15)
plt.xticks(np.arange(n_sample))
plt.ylabel('House price (USD)', fontsize=15)
plt.legend(fontsize=12)
plt.show()
```


    
![png](output_49_0.png)
    


# Class Implementation


```python
# Intro

Updated: 12/02/2024


Suppose you have built a fantastic machine learning model for predicting the selling price of a given house. Between the following two statements, what will make the biggest impression?

_'I predict that this house has a value of 450.000 euro'_

_or_

_'I predict that this house has a value between 435.000 euro and 465.000 euro with 90% certainty_'

While the boldness of the first statement might impress some people - certainly in the area of real estate - the later statement does convey information about the magnitude of uncertainty, which is incredibly useful. This is uncertainty quantification (UQ), for regression cases this comes down to constructing _prediction intervals_ (PI) and for classification to _prediction sets_.

Up until recently, the most common ways of uncertainty quantification were based on unrealistic assumptions (such as normality) amongst others and led to unsatisfactory results. Luckily, there is a 'new(er)' kid on the block called _conformal prediction_ or _conformal inference_, which is a very powerful and model-agnostic method to construct reliable prediction intervals or predictions sets without any distributional assumptions. 

A specific type of conformal prediction leverages the power of (conditional) quantile regression and provides a rather elegant way of constructing prediction intervals for regression cases that have nice properties, this will be the focus of this post.

# Conformal Quantile Regression: Concepts

## Background

Before diving straight into the specific type of conformal prediction of focus, let's quickly provide some more context about conformal prediction. Research in this field has been going on for some decades, mainly focused around the work of Vladimir Vovk and his colleagues.

Types of conformal prediction are often divided into 3 categories:
1. Full conformal prediction (the original implementation)
2. Split-conformal prediction
3. Cross-conformal prediction

Full conformal prediction, which is the original way of conducting conformal prediction following Vovk's research, requires vof refitting a very large amount of models, hence is computationally heavy. The latter two make use of data splitting that drive down the computation load substantially. 

## Conditional Quantile Regression and the Pinball Loss

For the case of regression with continuous outcomes, __(conditional) quantile regression__ is well known, but not particularly popular. While most regression models estimate the conditional mean, quantile regression aims to estimate a certain conditional quantile $q_{\tau}$ with $\tau \in [0, 1]$. The most common example would be to estimate the conditional $\tau=0.5$ quantile $q_{0.5}$, which is the conditional median. And yes indeed, one could estimate two conditional quantile models, one for lower bound and one for an upper bound to try to achieve valid prediction intervals. More on this a little later!

How can one estimate a conditional quantile instead of the (usual) conditional mean, you ask? Quantile regression achieves this by using a specific family of loss functions. Just like estimating the conditional mean is done by minimizing the mean squared error MSE loss, estimating conditional quantiles is achieved by minimizing the family of __pinball losses__ or _hockey stick losses_. Take the loss of the target and the estimated quantile using the features $\mathbf{x}$ as $q(\mathbf{x})$, for a conditional quantile $q_{\tau}$ with $\tau \in [0, 1]$

$$\rho_{\tau}(y, \hat{q}(\mathbf{x})) :=
\begin{cases}
    \tau(y-\hat{q}(\mathbf{x})), & \text{if } y-\hat{q}(\mathbf{x}) > 0\\
    (1-\tau)(y - \hat{q}(\mathbf{x})), & \text{otherwise.}
\end{cases}
$$

With some simplification by looking at the difference between target and predicted quantile as a kind of residual $\hat{\varepsilon} = y - \hat{q}(\mathbf{x})$ and taking the average, one can simplify the loss as:

__TODO IS THIS EQUIVALENT__

$$ \rho_{\tau}(y, \hat{q}(\mathbf{x})) := max(\tau \hat{\varepsilon}; (1 - \tau) \hat{\varepsilon})$$

And for all samples, taking the average as is usual:

$$\frac{1}{n} \sum_{i=1}^{n} max(\tau \hat{\varepsilon}_i; (1 - \tau) \hat{\varepsilon}_i)$$

In simple terms: this loss allows for incurring different losses for equal underprediction or overprediction magnitudes which can be seen by the differing angles to the left and the right. Also notice that if the prediction is equal to the target, the loss is 0, which is obviously a property that is desirable for a loss function.


![pinball_loss.png](attachment:pinball_loss.png)

To fit quantile regression models in practice, Gradient Boosting Machines (GBM) are a natural candidate. Instead of using a squared error loss, as is usual, one just uses the appropriate pinball loss.

As stated earlier, a potentially simple scheme for estimating prediction intervals arises almost naturally here: by estimating two conditional quantiles, a lower bound, $q_L$ ,and an upper bound $q_U$. 

The quantile levels $L, U \in [0, 1]$ to estimate are then chosen to achieve a certain _coverage_ $(1 - \alpha)$ and _mis-coverage_ $\alpha$ levels. Some examples:

* __A 90% symmetric prediction interval:__
    * mis-coverage is 10%:  $\alpha = 0.1$. 
    * coverage is 90%: $1 - \alpha = 0.9$
    * estimate conditional quantiles: $q_{0.05}(\mathbf{x})$ and $q_{0.95}(\mathbf{x})$, because miscoverage is distributed equally over lower and upper bounds
* __A 80% symmetric prediction interval:__
    * mis-coverage is 20%: , $\alpha = 0.2$. 
    * coverage is 80%: $1 - \alpha = 0.8$
    * estimate conditional quantiles for quantiles: $q_{0.1}(\mathbf{x})$ and $q_{0.9}(\mathbf{x})$
* __A 90% asymmetric prediction interval:__
    * mis-coverage is 10%:, $\alpha = 0.1$.
    * coverage is 90%:  $1 - \alpha = 0.9$
    * dsitribute mis-coverage for the lower and upper bound arises as desired, some examples:
        * 2.5% mis-coverage for the lower bound, 7.5% for the upper bound: estimate conditional quantile models for $q_{0.025}(\mathbf{x})$ and $q_{0.925}(\mathbf{x})$
        * 8% mis-coverage for the lower bound, 2% for the upper bound: estimate conditional quantile models for $q_{0.08}(\mathbf{x})$ and $q_{0.98}(\mathbf{x})$.


We could just stop here, because, estimating models by the appropriate pinball losses already guarantees convergence (statistcal consistency) to the population conditional quantiles, ... asymptotically $(n \to +\infty)$. In finite samples, it has been shown that the actual coverage is far off from the required (nominal) coverage. 

However, Conformalized conditional Quantile Regression (CQR), will take the main idea from quantile regression and augment it. This is what we will see next.

## (Non)-conformity Score

One of the most pivotal concepts in conformal prediction is the __(non)-conformity score $s_i$__. This score serves to encode (measure) the disagreement between predictions and targets. Many options exist on what kind of functional form of non-conformity score to choose. It can be as something simple as the absolute residual. However, since we are dealing with two conditional quantile regression models here, we need to apply some additional trickery

Suppose you have chosen two $\tau$-values, a lower and upper value $(L, U)$ (e.g. $L=0.05$ and $U=0.95$) for your two predicted conditional quantiles $\hat{q}_L(\mathbf{x}), \hat{q}_U(\mathbf{x})$ to construct a prediction interval. 

One elegant option that works well for this case is the following non-conformity score which can be calculated for each observation $i=1, ..., n$

$$s_i(y_i, \hat{q}(\mathbf{x}_i)) = \max\{\hat{q}_L(\mathbf{x}_i) - y ~;~ y - \hat{q}_U(\mathbf{x}_i)\}$$

In essence: for observations where $y_i$ falls within the prediction interval range, both values are negative and the distance to the closest boundary is taken as non-conformity score. For observations where the target falls outside the prediction interval: the distance to the closest boundary is taken.

At this point we have $n$ non-conformity scores $s_i$, or a vector $\mathbf{s}$. What to do with these? The idea is simple: __from this vector of $s_i$, take the (1 - $\alpha$)'th quantile and use this as a correction factor to adjust the predicted lower and upper conditional quantiles.__

$Quantile(s_1, ..., s_n; (1-\alpha))$

A slight modification is done to get more attractive finite sample properties.
 
$s_{adj} = Quantile(s_1, ..., s_n; (\frac{ceiling[(n + 1)(1-\alpha)]}{n}))$

Note that for very large $n$, this adjust value asymptotically will converge to the unadjusted (ordinary) sample quantile. Related is the fact that quantile regression can handle heteroskedastic noise, i.e. for certain regions, the uncertainty might be higher than for others (often this makes sense). 

Then, one simple thing needs to be done: adjust the resulting prediction interval using this calculated $s_{adj}$:


$$C(\mathbf{x}_i) = \left[ \hat{q}_{L}(\mathbf{x}_i) - s_{adj} ~ ; ~ \hat{q}_{U}(\mathbf{x}_i) + s_{adj} \right]$$


## Data Splitting Strategies: Split-Conformal Learning and Cross-Conformal Learning

Originally, full conformal learning, as was hinted before in the _Background_ section, led to a heavy computational load due to refitting $n$ models. Very similarly (and not by coincidence) to how data is often split to measure generalization performance, one can also choose to split data to conformalize or calibrate a models' predictions. Two flavours exist:

1. Split-Conformal learning
2. Cross-Conformal learning

Split conformal prediction divides all available data into two disjoint sets:
* $D_1$: data used for training, the proper training set to train a machine learning model on
* $D_2$: data used for conformalizing or calibrating the trained machine learning model to achieve proper coverage

While cross-conformal learning will divide the dataset into $K$ disjoint groups, and similarly to cross-validation, will have one of the groups play the role of the calibration set, while the others together form the proper training set and repeat this $K$ times until each datapoint has been part of the calibration set exactly once and $K-1$ times member of the proper training set. 

The reason why we apply data splitting in the first place is very similar to why measuring loss on the training set might be misleading: we don't want optimistically bias the (non-)conformity scores $s_i$ and hence the resulting adjustment factor. Such a bias would lead to worse results on unseen data, i.e. having incorrect coverages on unseen data.


# Conformal Quantile Regression in Python

Let's now see how we can use conformal quantile regression in Python. First we will load libraries and the _Ames Housing_ data and perform some basic preprocessing on it.

## Loading Libraries

# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pprint

# For reproduceability, showing which sklearn version is used. Use a modern version ideally >= 1.2
import sklearn; print(f"Scikit-learn version used: {sklearn.__version__}")

from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import (train_test_split)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import fetch_openml
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_pinball_loss

# Setting some parameters
plt.style.use("ggplot")  # Cleaner plotting

# Setting seed
seed = 42

## Loading Data

# Loading Ames housing data
housing = fetch_openml(name="house_prices", 
                       as_frame=True, 
                       parser='auto')

# Getting features (X) and target (y)
X = housing['data']
y = housing['target']

print(f"X (feature matrix) shape: {X.shape}")
print(f"y (target vector) shape: {y.shape}")

## Simple Preprocessing

# Quick look, appearantly, extremely skewed:
plt.hist(y, bins=50, color='salmon', alpha=0.8);
plt.title('Histogram of y  (house prices)')
plt.show();

plt.hist(np.log(y), bins=50, color='salmon', alpha=0.8);
plt.title('Histogram of log(y) ($log(house prices)$)')
plt.show();

# Maybe cut off the top? -- Check ECDF - Easier than Hist!...

# Cutting off the top 10%?
q_cutoff = 0.9
y_cutoff = np.quantile(a=y, q=q_cutoff)
plt.hist(y[y <= y_cutoff], bins=50, color='salmon', alpha=0.8);
plt.title('Histogram of y, cutoff at 90th quantile')
plt.show()

# Decision: let's cutoff at 90% to get some more nicely behaving distribution
remove_y_outliers = True
if remove_y_outliers:
    to_keep_idx = np.where(housing['target'] <= np.quantile(housing['target'], 0.9))
    y = housing['target'].loc[to_keep_idx]
    X = housing['data'].loc[to_keep_idx]

# Selecting some (mainly numeric) columns only
feat_cols = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
             'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr',
             'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF']
X = X[feat_cols].copy()

Let's split the data into 3 parts for now:
* $D_1$: the _proper training set_ used for training the model. This part will be also divided into parts using cross-validation for tuning hyperparameters. (60% of the observations)
* $D_2$: the _calibration set_ used to conformalize the prediction intervals. (50% of 40% = 20% of observations)
* $D_3$: a completely unseen test set that can be used to see if the coverages hold in unseen data. (50% of 40% = 20% of observations)

# Splitting in train-calibration-test set
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=seed)
X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, shuffle=True, random_state=seed)

print(f'Sample sizes for train set: {len(y_train)} \n Sample size for calibration set: {len(y_cal)} \n Sample size for test set: {len(y_test)}')

## Training and Tuning Conditional Quantile Regression GBMs

Let's use a very general approach for training and tuning GBM models, with the big differences being:
* Choosing a coverage level $1 - \alpha$, and choosing how to allocate the mis-coverage over the lower and upper boundaries. We will choose a symmetric 90% prediction interval, hence: $q_L = q_{0.05}$ and $q_U = q_{0.95}$.
* Training and tuning of the two conditional quantile models instead of a single (conditional mean) model.
* Use of the appropriate pinball losses, both for training and tuning. 

Except this, the procedure we will follow is quite common:

1. Defining a hyperparameter grid. We'll take some basic hyperparameters and define reasonable values for them. We will use the same grid for both conditional quantiles we estimate.
2. Define scoring metrics for each model using `make_scorer()`, for conditional mean models you'd usually just use default scoring.
3. Instantiate a GBM for each quantile to estimate. We will use the newer `HistGradientBoostingRegressor` that takes inspiration from XGBoost and LightGBM implementations, in recent versions this supports estimation of quantiles by defining `loss=quantile` and supplying the requested $\tau$ to the `quantile` argument.
4. Define a data splitting strategy, for this we will use a repeated K-fold cross-validation setup using the built-in `RepeatedKFold` class. We'll use 2-times repeated 4-fold cross-validation.
5. Search the hyperparameter grid by sampling using random search (`RandomizedSearchCV`), we will only search 10 values, for simplicity sake.
6. Obtain the optimal parameters from the 

# Generating hyperparameter grid
param_grid = {'learning_rate': [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
              'max_iter': [50, 100, 150, 250],  # amount of trees
              'max_depth': [2, 3, 5, 7],
              'min_samples_leaf': [3, 5, 10, 20, 30]}        

# Defining mis-coverage level alpha
alpha = 0.10

# Defining tau to predict the quantile q_tau for
taus = [0.05, 0.95]  # contains (L, U) levels

# Setting some other CV params
n_hyperparams = 20
n_folds = 4
n_repeats = 2
seed = 42

# Create the appropriate pinball loss functions
pinball_losses = [make_scorer(mean_pinball_loss, alpha=tau, greater_is_better=False) for tau in taus]
pinball_losses = dict(zip(taus, pinball_losses))  # Dictionary with the alphas as keys and losses as values

# Cross-validation with chosen search strategy for hyperparameter values
models = dict.fromkeys(taus, None)  # Dictionary with keys being the tau values to save models in
tuning_objects = dict.fromkeys(taus, None)  # Same, but to save tuned model in
for tau in taus:
    print(f'Tuning hyperparameters for conditional quantile regression model: q_{tau}')
    models[tau] = HistGradientBoostingRegressor(loss='quantile', quantile=tau, random_state=seed) 
    # From sklearn v1.1 onwards you can use faster HistGradientBoostingRegressor instead of GradientBoostingRegressor
    
    # Repeated K-fold cross-validation data resampling
    cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    
    # Tuning using random grid search
    tuning_objects[tau] = RandomizedSearchCV(estimator=models[tau],
                                             cv=cv,
                                             param_distributions=param_grid,
                                             n_iter=n_hyperparams,
                                             scoring=pinball_losses[tau],
                                             random_state=seed,
                                             verbose=1)
    tuning_objects[tau].fit(X=X_train, y=y_train)
    print(f'Optimal hyperparameters {tuning_objects[tau].best_params_}')
    print('*' * 125)

Let's now use the optimally (re)-trained models for the lower and upper quantiles and predict instances on the (unseen) calibration data.

# Gather predictions on calibration data, putting them in a N x 2 NumPy array
lower_preds_cal = tuning_objects[taus[0]].predict(X=X_cal)
upper_preds_cal = tuning_objects[taus[1]].predict(X=X_cal)
preds_cal = np.vstack([lower_preds_cal, upper_preds_cal]).T  # Transpose to get N x 2 instead of 2 x N

lower_preds_test = tuning_objects[taus[0]].predict(X=X_test)
upper_preds_test = tuning_objects[taus[1]].predict(X=X_test)
preds_test = np.vstack([lower_preds_test, upper_preds_test]).T

# Make them into Pandas DataFrames with correct indices (predict returns NumPy arrays...)
preds_cal_pd = pd.DataFrame(preds_cal, index=X_cal.index, columns=['lower', 'upper'])
preds_test_pd = pd.DataFrame(preds_test, index=X_test.index, columns=['lower', 'upper'])

## Conformalizing the Quantile Regression Models

We take the predictions on the calibration dataset ($D_2$) and translate the theory into code. For this we will make a small function `calculate_conformal_correction`.

def calculate_conformal_correction(preds_cal, target_cal, alpha):
    """
    Calculates conformity scores and calculates the modified quantile to correct the raw prediction intervals
    provided to achieve correct requested coverage and mis-coverage (alpha). Ideally this is performed on a 
    separate calibration dataset (hence pred_cal, target_cal).
    """
    # Calculate the 'deviations' from lower and upper predicted quantiles
    preds_lower = preds_cal[:, 0]  # First col is lower bound
    preds_upper = preds_cal[:, 1]  # Second col is upper bound
    dev = np.vstack([preds_lower - target_cal, target_cal - preds_upper])
    nonconf_scores = np.max(dev, axis=0)  # max of each 2-pair of deviations: i=1, ..., N
    
    # Take the 'modified' quantile of these non-conformity scores 
    n = len(target_cal)
    alpha_compl_mod = np.ceil((n + 1) * (1 - alpha)) / n
    correction = np.quantile(a=nonconf_scores, q=alpha_compl_mod)
    
    return correction

# Use function to calculate the correction factor
conformal_correction = calculate_conformal_correction(preds_cal=preds_cal,
                                                      target_cal=y_cal,
                                                      alpha=0.1)
print(f"The conformal correction calculated on the calibration data is: {conformal_correction: .2f} (USD)")

# Compare the correction magnitude to average of target in calibration set to get relative idea
print(f"The conformal correction relative to the average house price: {conformal_correction / y_cal.mean() * 100:.2f}%")

The calculated correction factor to adjust both lower and upper (raw) prediction bounds is (+) 6039 (USD). This is approximately 3.82% of the average house, hence nothing dramatic. This seems very reasonable compared to the magnitude of house prices. 

We can now progress by just applying the adjustment, since it's positive it means that we will enlarge the prediction interval, hence moving the lower bound down and the upper bound up. 

# Conformalize the calibration and test predictions using the calculated correction
lower_preds_cal_conformal = lower_preds_cal - conformal_correction
upper_preds_cal_conformal = upper_preds_cal + conformal_correction

lower_preds_test_conformal = lower_preds_test - conformal_correction
upper_preds_test_conformal = upper_preds_test + conformal_correction

## Conforming Coverage

Let's now check, on the unseen test data set, if the requested coverage is achieved. We make a very simple function to 

def calculate_coverage(preds, target):
    """
    Calculates if the target falls within the lower and upper bounds to calculate actual coverage.
    """
    # Count observations where target is within bounds (1 if yes, 0 if not)
    target_in_bounds_indicator = np.where((preds[:, 0] <= target) & (preds[:, 1] > target), 1, 0)
    
    # Return average (=proportion with the 0/1 from above)
    return np.mean(target_in_bounds_indicator)

# Use function for the raw predictions on test set and the conformalized ones
preds_test_conformal = np.vstack([lower_preds_test_conformal, upper_preds_test_conformal]).T

coverage_test_raw = calculate_coverage(preds=preds_test, target=y_test)
coverage_test_conformal = calculate_coverage(preds=preds_test_conformal, target=y_test)

print(f"** Requested coverage levels: {1 - alpha: .2f} **")
print(f"Actual coverage on test set without conformal correction: {coverage_test_raw: .2f}")
print(f"Actual coverage on test set with conformal correction: {coverage_test_conformal: .2f}")

We see that, using the correction calculated on the separate calibration set, also leads to getting nearer to requested coverage on unseen data. We can also performan a sanity check that on the calibration set itself, the correction achieves (near) perfect coverage.

# Sanity check: coverage on the calibration set itself should be near perfect
preds_cal_conformal = np.vstack([lower_preds_cal_conformal, upper_preds_cal_conformal]).T

coverage_cal_raw = calculate_coverage(preds=preds_cal, target=y_cal)
coverage_cal_conformal = calculate_coverage(preds=preds_cal_conformal, target=y_cal)

print(f"** Requested coverage levels: {1 - alpha: .2f} **")
print(f"Actual coverage on calibration set without conformal correction: {coverage_cal_raw: .2f}")
print(f"Actual coverage on calibration set without conformal correction: {coverage_cal_conformal: .2f}")

Yes it does work as expected: going from 80% coverage without conformalizing to 90% as was requested!

def create_conformal_summary_df(preds_cal, preds_test, target_cal, target_test, alpha, alpha_lower=None):
    """
    Create both symmetric and asymmetric conformalized preds and make an overview DF including coverage.
    
    """
    
    preds_c_symmetric = conformalize_preds(preds_cal=preds_cal, 
                                           preds_test=preds_test, 
                                           target_cal=target_cal,
                                           alpha=alpha,
                                           mode='quantile')
    preds_c_asymmetric = conformalize_preds(preds_cal=preds_cal, 
                                            preds_test=preds_test,
                                            target_cal=target_cal,
                                            alpha=alpha,
                                            alpha_lower=alpha_lower,
                                            mode='quantile_asymmetric')
    
    # Making overview DataFrame
    df = pd.DataFrame({'lower_pred_test_q': lower_preds_test, 
                       'upper_pred_test_q': upper_preds_test})
    
    df['y_test'] = target_test
    df['lower_pred_test_symm'] = preds_c_symmetric[:, 0]
    df['upper_pred_test_symm'] = preds_c_symmetric[:, 1]
    df['lower_pred_test_asymm'] = preds_c_asymmetric[:, 0]
    df['upper_pred_test_asymm'] = preds_c_asymmetric[:, 1]
    df['coverage_q'] = np.where((df['lower_pred_test_q'] <= df['y_test']) & (df['y_test'] <= df['upper_pred_test_q']), 1, 0) # Correct?
    df['coverage_symm'] = np.where((df['lower_pred_test_symm'] <= df['y_test']) & (df['y_test'] <= df['upper_pred_test_symm']), 1, 0)
    df['coverage_asymm'] = np.where((df['lower_pred_test_asymm'] <= df['y_test']) & (df['y_test'] <= df['upper_pred_test_asymm']), 1, 0)
    df['width_q'] = np.abs(df['upper_pred_test_q'] - df['lower_pred_test_q'])
    df['width_symm'] = np.abs(df['upper_pred_test_symm'] - df['lower_pred_test_symm'])
    df['width_asymm'] = np.abs(df['upper_pred_test_asymm'] - df['lower_pred_test_asymm'])

    # Printing some diagnostics
    print(f"Coverage for original quantile predictions is: {df['coverage_q'].mean():.3f}")
    print(f"Coverage for symmetric conformalized predictions is: {df['coverage_symm'].mean():.3f}")
    print(f"Coverage for asymmetric conformalized predictions is: {df['coverage_asymm'].mean():.3f}")

    return df

df_summary = create_conformal_summary_df(preds_cal=[lower_preds_cal, upper_preds_cal],
                                         preds_test=[lower_preds_test, upper_preds_test],
                                         target_cal=y_cal, 
                                         target_test=y_test,
                                         alpha=0.1,
                                         alpha_lower=0.05)
print('Showing 5 top rows of the predictions with various bounds...')
display(df_summary.head())

df_summary = create_conformal_summary_df(preds_cal=[lower_preds_cal, upper_preds_cal],
                                         preds_test=[lower_preds_test, upper_preds_test],
                                         target_cal=y_cal, 
                                         target_test=y_test,
                                         alpha=0.1)
df_summary.head()

## Visualization

Let's try to get an intuitive visualization of what is happening by taking a selection of observations (e.g. 100) and plotting them with their associated raw prediction intervals straight from the conditional quantile regression and the conformalized equivalents.

# Let's put everything in a DF first
df_summary = pd.DataFrame(data={
    'y_test': y_test,
    'lower_preds_test': lower_preds_test,
    'upper_preds_test': upper_preds_test,
    'lower_preds_test_conformal': lower_preds_test_conformal,
    'upper_preds_test_conformal': upper_preds_test_conformal
}, index=y_test.index)

# Check results
df_summary.head(5)

# Plotting sample of some observations
# First, sample some observations
n_sample = 10
preds_test_sample = df_summary.sample(n_sample, random_state=seed).reset_index()  # reset_index, it makes plot cleaner

# Plotting
plt.figure(figsize=(20, 10))

# Plotting the PIs with vertical lines and whiskers
# For raw prediction intervals
plt.errorbar(x=preds_test_sample.index - 0.01, # small adjustment to not have the 2 bars overlap perfectly
             y=(preds_test_sample['lower_preds_test'] + preds_test_sample['upper_preds_test']) / 2,  # "midpoint of the PI"
             yerr=[(preds_test_sample['upper_preds_test'] - preds_test_sample['lower_preds_test']) / 2],  # half the width around the point
             fmt=' ', color='lightblue', ecolor='cadetblue', elinewidth=3, capsize=5, capthick=2, alpha=0.7,
             label='Raw Prediction Interval')

# For conformal prediction intervals
plt.errorbar(x=preds_test_sample.index + 0.01, # small adjustment to not have the 2 bars overlap perfectly
             y=(preds_test_sample['lower_preds_test_conformal'] + preds_test_sample['upper_preds_test_conformal']) / 2,  # "midpoint of the PI"
             yerr=[(preds_test_sample['upper_preds_test_conformal'] - preds_test_sample['lower_preds_test_conformal']) / 2],  # half the width around the point
             fmt=' ', color='lightcoral', ecolor='salmon', elinewidth=3, capsize=5, capthick=2, alpha=0.7,
             label='Conformal Prediction Interval')

# Plot true target values
plt.plot(preds_test_sample.index, preds_test_sample['y_test'], 'o', markersize=10, color='darkslategray', label='True target value')
plt.title(f"Sample of {n_sample} houses and their raw PI and conformalized PI", fontsize=15)
plt.xlabel('House', fontsize=15)
plt.xticks(np.arange(n_sample))
plt.ylabel('House price (USD)', fontsize=15)
plt.legend(fontsize=12)
plt.show()

# Class Implementation
```
