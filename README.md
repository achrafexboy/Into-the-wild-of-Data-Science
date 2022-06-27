# A Brief presentation about API processing

Data preprocessing is the art/science of converting data to the best way possible, which involve an elegant blend of domain expertise, intuition and mathematics.


## 1. Data Exploration

### 1.1 Variables

**Definition**: any measurable property/characteristic of a phenomenon being observed. They are called 'features' because the value they take may vary (and it usually does) in a population. 

**Types of Variable**

| Type        | Sub-type   | Definition                                                   | Example                        |
| ----------- | ---------- | ------------------------------------------------------------ | ------------------------------ |
| Categorical | Nominal    | Variables with values selected from a group of categories, while not having any kind of natural order. | Gender, car types              |
|             | Ordinal    | A categorical variable whose categories can be meaningfully ordered. | Grade of an exam               |
| Numerical   | Discrete   | Variables whose values are either finite or countably infinite. | Number of children in a family |
|             | Continuous | Variable which can take on infinitely many, uncountable values. | House prices, time passed      |



### 1.2 Variable Identification

**Definition**: Identify the data types of each variable.

**Note**:  In reality we may have mixed type of variable for a variety of reasons.



### 1.3 Univariate Analysis

Descriptive statistics on one single variable.

| Variable    | What to look for                                             |
| ----------- | ------------------------------------------------------------ |
| Categorical | **Shape**: Histogram/ Frequency table...                     |
| Numerical   | **Central Tendency** :  Mean/ Median/ Mode<br>**Dispersion** :  Min/ Max/ Range/ Quantile/ IQR/ MAD/ Variance/ Standard Deviation/ <br>**Shape** : Skewness/ Histogram/ Boxplot... |

Below are some methods that can give us the basic stats on the variable:

```python
pandas.Dataframe.describe()
```
```python
 pandas.Dataframe.dtypes
```
- Barplot
- Countplot
- Boxplot
- Distplot



### 1.4 Bi-variate Analysis

Descriptive statistics between two or more variables.

- Scatter Plot
- Correlation Plot
- Heat Map

**Scatter Plot** is a type of plot or mathematical diagram using Cartesian coordinates to display values for typically two variables for a set of data. If the pattern of dots slopes from lower left to upper right, it indicates a positive correlation between the variables being studied. If the pattern of dots slopes from upper left to lower right, it indicates a negative correlation.

**Correlation plot** can be used to quickly find insights. It is used to investigate the dependence between multiple variables at the same time and to highlight the most correlated variables in a data table.

**Heat map** (or heatmap) is a graphical representation of data where the individual values contained in a matrix are represented as colors.



## 2. Data Cleaning

### 2.1 Missing Values

**Definition**: no value is stored in a certain observation within a variable.

#### 2.1.1 Why Missing Data Matters

- certain algorithms cannot work when missing value are present
- even for algorithm that handle missing data, without treatment the model can lead to inaccurate conclusion



#### 2.1.2 How to Handle Missing Data

| Method                         | Definition                                                   | Pros                                                         | Cons                                                         |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Listwise Deletion              | excluding all cases (listwise) that have missing values      | preserve distribution if MCAR (missing completely at random) | 1. may discard too much data and hurt the model<br>2. may yield biased estimates if not MCAR (as we keep a special subsample from the population) |
| Mean/Median/Mode Imputation    | replacing the NA by mean/median/most frequent values (for categorical feature) of that variable | good practice if MCAR                                        | 1. distort distribution<br>2. distort relationship with other variables |
| End of distribution Imputation | replacing the NA by values that are at the far end of the distribution of that variable, calculated by mean + 3*std | Captures the importance of missingness if there is one       | 1. distort distribution<br />2. may be considered outlier if NA is few or mask true outlier if NA is many.<br />3. if missingness is not important this may mask the predictive power of the original variable |
| Random Imputation              | replacing the NA by taking a random value from the pool of available observations of that variable | preserve distribution if MCAR                                | not recommended in business settings for its randomness (different result for same input) |
| Arbitrary Value Imputation     | replacing the NA by arbitrary values                         | Captures the importance of missingness if there is one       | 1. distort distribution<br />2. typical used value: -9999/9999. But be aware it may be regarded as outliers. |
| Add a variable to denote NA    | creating an additional variable indicating whether the data was missing for that observation | Captures the importance of missingness if there is one       | expand feature space                                         |

In real settings, when it's hard to decide the missing mechanism or there's few time to study deeply about each missing variables, the popular way is to adopt:

- Mean/Median/Mode Imputation (depend on the distribution)
- End of distribution Imputation
- Add a variable to denote NA

simultaneously, so that we both catch the value of missingness and obtain a complete dataset.

**Note**: Some algorithms like XGboost incorporate missing data treatment into its model building process, so you don't need to do the step. However it's important to make sure you understand how the algorithm treat them and explain to the business team.



### 2.2 Outliers

**Definition**:  An outlier is an observation which deviates so much from the other observations as to arouse suspicions that it was generated by a different mechanism. 

#### 2.2.1 Why Outlier Matters

The presence of outliers may:

- make algorithm not work properly
- introduce noises to dataset
- make samples less representative

#### 2.2.2 Outlier Detection

All the methods here listed are for univariate outlier detection.

| Method                           | Definition                                                   | Pros                                                         | Cons                                                         |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Detect by arbitrary boundary     | identify outliers based on arbitrary boundaries              | flexible                                                     | require business understanding                               |
| Mean & Standard Deviation method | outlier detection by Mean & Standard Deviation Method        | good for variable with Gaussian distribution (68-95-99 rule) | sensitive to extreme value itself (as the outlier increase the std) |
| IQR method                       | outlier detection by Interquartile Ranges Rule               | robust than Mean & SD method as it use quantile & IQR. Resilient to extremes. | can be too aggressive                                        |
| MAD method                       | outlier detection by Median and Median Absolute Deviation Method | robust than Mean & SD method. Resilient to extremes.         | can be too aggressive                                        |

However, beyond these methods, it's more important to keep in mind that the business context should govern how you define and react to these outliers. The meanings of your findings should be dictated by the underlying context, rather than the number itself.



#### 2.2.3 How to Handle Outliers

| Method                          | Definition                                                   | Pros                             | Cons                                        |
| ------------------------------- | ------------------------------------------------------------ | -------------------------------- | ------------------------------------------- |
| Mean/Median/Mode Imputation     | replacing the outlier by mean/median/most frequent values of that variable | preserve distribution            | lose information of outlier if there is one |
| Discretization                  | transform continuous variables into discrete variables       | minimize the impact from outlier | lose information of outlier if there is one |
| Imputation with arbitrary value | impute outliers with arbitrary value.                        | flexible                         | hard to decide the value                    |
| Windsorization                  | top-coding & bottom coding (capping the maximum of a distribution at an arbitrarily set value, vice versa). | prevent model over-fitting       | distort distribution                        |
| Discard outliers                | drop all the observations that are outliers                  | depends                          | lose information of outlier if there is one |

There are many strategies for dealing with outliers in data, and depending on the context and data set, any could be the right or the wrong way. It’s important to investigate the nature of the outlier before deciding.



## 3. Data transformation

### 3.1 Feature Scaling

**Definition**: Feature scaling is a method used to standardize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

#### 3.1.1 Why Feature Scaling Matters

- If range of inputs varies, in some algorithms, object functions will not work properly.

- **Gradient descent** converges much faster with feature scaling done. Gradient descent is a common optimization algorithm used in logistic regression, SVMs,  neural networks etc.

- Algorithms that involve **distance calculation** like KNN, Clustering are also affected by the magnitude of the feature. Just consider how Euclidean distance is calculated: taking the square root of the sum of the squared differences between observations. This distance can be greatly affected by differences in scale among the variables. Variables with large variances have a larger effect on this measure than variables with small variances.

#### 3.1.2 How to Handle Feature Scaling

| Method                                            | Definition                                                   | Pros                                                         | Cons                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Normalization - Standardization (Z-score scaling) | removes the mean and scales the data to unit variance.<br/>z = (X - X.mean) /  std | feature is rescaled to have a standard normal distribution that centered around 0 with SD of 1 | compress the observations in the narrow range if the variable is skewed or has outliers, thus impair the predictive power. |
| Min-Max scaling                                   | transforms features by scaling each feature to a given range. Default to [0,1].<br/>X_scaled = (X - X.min / (X.max - X.min) | /                                                            | compress the observations in the narrow range if the variable is skewed or has outliers, thus impair the predictive power. |
| Robust scaling                                    | removes the median and scales the data according to the quantile range (defaults to IQR)<br/>X_scaled = (X - X.median) / IQR | better at preserving the spread of the variable after transformation for skewed variables | /                                                            |



As we can see, Normalization - Standardization and Min-Max method will compress most data to a narrow range, while robust scaler does a better job at keeping the spread of the data, although it cannot **remove** the outlier from the processed result.



### 3.2 Feature Transformation

#### 3.2.1 How to Handle Feature Transformation

| Method                     | Definition                                               |
| -------------------------- | -------------------------------------------------------- |
| Logarithmic transformation | log(x+1).  We use (x+1) instead of x to avoid value of 0 |
| Reciprocal transformation  | 1/x. Warning that x should not be 0.                     |
| Square root transformation | x**(1/2)                                                 |
| Exponential transformation | X**(m)                                                   |
| Box-cox transformation[12] | (X**λ-1)/λ                                               |
| Quantile transformation    | transform features using quantiles information           |



## 4. Feature Selection

**Definition**:  Feature Selection is the process of selecting a subset of relevant features for use in machine learning model building. 

It is not always the truth that the more data, the better the result will be. Including irrelevant features (the ones that are just unhelpful to the prediction) and redundant features (irrelevant in the presence of others) will only make the learning process overwhelmed and easy to cause overfitting.

With feature selection, we can have:

- simplification of models to make them easier to interpret
- shorter training times and lesser computational cost
- lesser cost in data collection
- avoid the curse of dimensionality
- enhanced generalization by reducing overfitting 

We should keep in mind that different feature subsets render optimal performance for different algorithms. So it's not a separate process along with the machine learning model training. Therefore, if we are selecting features for a linear model, it is better to use selection procedures targeted to those models, like importance by regression coefficient .



### 4.1 Filter Method

Filter methods select features based on a performance measure regardless of the ML algorithm later employed.

Univariate filters evaluate and rank a single feature according to a certain criteria, while multivariate filters evaluate the entire feature space. Filter methods are:

- selecting variable regardless of the model
- less computationally expensive
- usually give lower prediction performance

As a result, filter methods are suited for a first step quick screen and removal of irrelevant features.

| Method                    | Definition                                                   |
| ------------------------- | ------------------------------------------------------------ |
| Variance                  | removing features that show the same value for the majority/all of the observations (constant/quasi-constant features) |
| Correlation               | remove features that are highly correlated with each other   |
| Chi-Square                | Compute chi-squared stats between each non-negative feature and class |
| Mutual Information Filter | Mutual information measures how much information the presence/absence of a feature contributes to making the correct prediction on Y. |
| Univariate ROC-AUC or MSE | builds one decision tree per feature, to predict the target, then make predictions and ranks the features according to the machine learning metric (roc-auc or mse) |

### 4.2 Dimensionality Reduction

- PCA projection

- LDA projection





------

The end, by : 

* ACHRAF FAYTOUT
* MOHAMMED AMRANI ALAOUI
