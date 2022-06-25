# Scientific Libraries
import numpy as np
import pandas as pd
import seaborn as sns
# scikit-learn lib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Random Gen
import random
# Get the data
titanic_data = sns.load_dataset('titanic')
titanic_data.head()
df1 = titanic_data.copy()


# Methode class

class MissingValues:

    # init function
    def __init__(self, dataFrame, methods = None, rate = 0.1, rateOpt = 0.1):   # default rate 0.1
        self.rate = rate 
        self.dataFrame = dataFrame
        self.methods = methods
        self.rateOpt = rateOpt

        # Get the missing numerical columns
        self.df_numeric = dataFrame.select_dtypes(include=[np.number]) # select numerical columns from the dataFrame
        self.numeric_cols = self.df_numeric.columns.values # labels of numerical columns

        self.missing_numeric_columns = []
        for column in self.df_numeric.columns:
            if self.df_numeric[column].isnull().mean() > self.rate: # if the percentage of the missing values in a column is greater than %rate
                self.missing_numeric_columns.append(column)         # ... consider it as a missing numerical column
        

        # Get the missing categorical columns
        self.df_categorical = dataFrame.select_dtypes(exclude=[np.number]) # select categorical columns from the dataFrame
        self.categorical_cols = self.df_categorical.columns.values # labels of categorical columns

        self.missing_categorical_columns = []
        for column in self.df_categorical.columns:
            if self.df_categorical[column].isnull().mean() > self.rate: # if the percentage of the categorical values in a column is greater than %rate
                self.missing_categorical_columns.append(column)         # ... consider it as a categorical numerical column

        self.missing_values_columns = []
        for column in self.dataFrame.columns:
            if self.dataFrame[column].isnull().mean() > self.rateOpt: # if the percentage of the missing values in a column is greater than %rate
                self.missing_values_columns.append(column)         # ... consider it as a missing numerical column

        self.all_missing_columns = self.missing_numeric_columns + self.missing_categorical_columns # get all the missing column (num, categ)
    # End init function

    ## Droping Rows or columns
    # Row Imputation methode
    def impute_nan_row(self):
        self.df_copy = self.dataFrame.copy()
        self.df_copy = self.df_copy.dropna(axis = 0) # Remove the rows with missing values
        return self.df_copy
    # End row Imputation methode

    # Column Imputation methode
    def impute_nan_column(self):
        self.df_copy = self.dataFrame.copy()
        self.df_copy = self.df_copy.drop(
            columns=self.missing_values_columns, axis = 1) # Remove all columns with missing values
        return self.df_copy
    # End column Imputation methode

    ## Filling Methods [Numerical]
    # Mean methode
    def impute_nan_mean(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_numeric_columns:
            mean = self.df_copy[column].mean() # Calculate the mean value of each column
            self.df_copy[column].fillna(mean, inplace = True) # Fill the missing value with the value calculated
        return self.df_copy
    # End Mean methode

    # median methode
    def impute_nan_median(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_numeric_columns:
            median = self.df_copy[column].median() # Calculate the median value of each column
            self.df_copy[column].fillna(median, inplace = True) # Fill the missing value with the value calculated
        return self.df_copy
    # End median methode

    # End of Distribution Imputation methode
    # ... replace missing data with values that are at the tails of the distribution of the column (variable).
    def impute_nan_eod(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_numeric_columns:
            eod_value = self.df_copy[column].mean() + 3 * self.df_copy[column].std() # Calculate the eod value of each column
            self.df_copy[column].fillna(eod_value, inplace = True) # Fill the missing value with the value calculated
        return self.df_copy
    # End "End of Distribution Imputation" methode

    # Arbitrary Value Imputation methode
    def impute_nan_arbitrary_num(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_numeric_columns:
            rdm = np.nan
            while rdm == np.nan : # except if the column have all values Nan [drop it]
                # random integer from 0 to random.randint(0, self.df_copy.shape[0]) ~> pass as parameter to iloc
                rdm = self.df_copy[column].iloc[random.randint(0, self.df_copy.shape[0])]
            self.df_copy[column].fillna(rdm, inplace=True)
        return self.df_copy
    # End Arbitrary Value Imputation methode

    # mode methode
    def impute_nan_mode(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_categorical_columns:
            self.df_copy[column].fillna(
                self.df_copy[column].mode().iloc[0], inplace=True) 
        return self.df_copy
    # End mode methode

    # Arbitrary value columns methode
    def impute_nan_arbitrary_cat(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_categorical_columns:
            rdm = np.nan
            while rdm == np.nan : # except if the column have all values Nan [drop it]
                # random integer from 0 to random.randint(0, self.df_copy.shape[0])  ~> pass as parameter to iloc
                rdm = self.df_copy[column].iloc[random.randint(0, self.df_copy.shape[0])]
            self.df_copy[column].fillna(rdm, inplace=True)
        return self.df_copy
    # End Arbitrary value columns methode

    def arb_value(self, max):
        if(max < 10):
            return 9
        elif(max < 100):
            return 99
        elif(max < 1000):
            return 999
        elif(max < 10000):
            return 9999
        elif(max < 100000):
            return 99999
        elif(max < 1000000):
            return 999999
        elif(max < 10000000):
            return 9999999


def method_chosing(classTest):
    for method in classTest.methods:
        print("====method====")
        print(method)
        if(method == "row"):
            classTest.dataFrame = classTest.impute_nan_row()
        elif(method == "column"):
            classTest.dataFrame = classTest.impute_nan_column()
        elif(method == "mean"):
            classTest.dataFrame = classTest.impute_nan_mean()
        elif(method == "median"):
            classTest.dataFrame = classTest.impute_nan_median()
        elif(method == "eod"):
            classTest.dataFrame = classTest.impute_nan_eod()
        elif(method == "arbitrary"):
            classTest.dataFrame = classTest.impute_nan_arbitrary_value()
        elif(method == "mode"):
            classTest.dataFrame = classTest.impute_nan_mode()
        elif(method == "arbitraryCat"):
            classTest.dataFrame = classTest.impute_nan_arbitrary_columns()
    return classTest.dataFrame


# print("type: ", type(df.isnull().mean()))
# print(df.isnull().mean()[0])
results = df1.isnull().mean().to_dict()

# print("type: ", type(resultDict))

# print("dict: ", resultDict)

# for key in resultDict:
#   print(key, '->', resultDict[key])


# results = df.isnull().mean()
# data = ""


"""
************* Data Reduction *************
"""


class DataReduction:

    # init function
    def __init__(self, dataFrame, method = "pca", target = None, n_components = 2): # default values "pca" & number of compenents if it's not defined 
        self.df = dataFrame
        self.method = method
        self.target = target
        self.n_components = n_components
        self.Y = self.df[target]
        self.df = self.df.drop([target], axis = 1) # drop the target from df
        self.df_numeric = self.df.select_dtypes(include = [np.number]) # Suppose that the data have already passed by Cleaning 
        self.df_categ = self.df.select_dtypes(exclude = [np.number]) # ... same thing go with categorical data
    # End init function

    # start Row PCA methode
    def pca(self):
        pca = PCA(n_components = self.n_components)
        # fitting pca with numerical columns ≠ target
        X_trans = pca.fit_transform(self.df_numeric)

        # creating a list of column names PC1, PC2 ... 
        column_values = [f'PC{(i+1)}' for i in range(X_trans.shape[1])] 

        self.df1 = pd.DataFrame(data = X_trans, columns = column_values) # convert it to a data frame
        self.df1 = pd.concat([self.df1, self.df_categ, self.Y], axis=1) # merge it with categorical columns that have not been processed & target

        return self.df1
    # End row PCA methode

    # start factor analysis methode
    def fa(self):
        fa = FactorAnalysis(n_components = self.n_components, random_state = 0)
        # fitting fa with numerical columns ≠ target
        X_transformed = fa.fit_transform(self.df_numeric)

        # creating a list of column names Factor1 , Factor2 ...
        column_values = [f'Factor{(i+1)}' for i in range(X_transformed.shape[1])]

        self.df1 = pd.DataFrame(data = X_transformed, columns = column_values) # convert it to a data frame
        self.df1 = pd.concat([self.df1, self.df_categ, self.Y], axis=1) # merge it with categorical columns that have not been processed & target

        return self.df1


def method_chosing_data_red(classTest):
    for method in classTest.method:
        print("====method====")
        print(method)
        if(method == "pca"):
            classTest.df = classTest.pca()
        if(method == "fa"):
            classTest.df = classTest.fa()
    return classTest.df1


# Using exemple :

"""
classTest = DataReduction(df, ["pca"], "alive", n_components)
new_df = method_chosing_data_red(classTest)
"""


"""
************* FeatureSelection *************
"""

class FeatureSelection:

    # init function
    def __init__(self, dataFrame, target):
        self.df = dataFrame
        self.target = target

        self.df_numeric = self.df.select_dtypes('number')
        self.df_categ = self.df.select_dtypes(exclude = [np.number])
        self.Y = self.df[self.target]

        if self.target in self.df_categ :
            le = LabelEncoder()
            self.Y = le.fit_transform(np.array(self.Y))
            self.Y = pd.DataFrame(self.Y, columns = [target])
        
        self.df.drop([self.target], axis = 1)
    # End init function

    # start Information gain methode
    def mic(self, rate = 0.1):
        X = self.df_numeric
        y = np.ravel(np.array(self.Y))

        mi_score = MIC(X, np.ravel(np.array(y)))
        # return table with the index of the features that meets the criteria
        # ... if X and y are dependent with a Mic higher then a specific rate
        mi_score_selected_index = np.where(mi_score > rate)[0]

        # The features with the MIC score > rate
        X = X.iloc[:, mi_score_selected_index]

        self.df1 = pd.concat([X, self.df_categ, self.Y], axis = 1) # merge it with categorical columns that have not been processed & target

        return self.df1
    # End Information gain methode

    # start SelectK methode
    def SelectK(self, sc_f = chi2, k = "all"):
        X = self.df_numeric
        y = np.ravel(np.array(self.Y))

        # "k" features with highest chi-squared statistics score are selected
        # SelectKbest can be used with other methods not necessary chi2
        new_features = SelectKBest(sc_f, k = k)
        new_features.fit(X, y)
        cols = new_features.get_support(indices = True)  # Columns names indexs
        X_kbest_features = X.iloc[:, cols]

        self.df1 = pd.concat([X_kbest_features, self.df_categ, self.Y], axis = 1)
        return self.df1
    # End SelectK methode

    # start Forward feature selection methode
    def ffs(self,  type = 'Regression',  n_features = "auto"):
        X = self.df_numeric
        y = np.ravel(np.array(self.Y))
        if type == 'Regression' :
            model = LinearRegression()
        elif type == 'Classification' :
            model =  LogisticRegression()
        
        sfs = SequentialFeatureSelector(model, n_features_to_select = n_features)
        sfs.fit(X, y)

        # the best features (features is an array of true and false )
        features = sfs.get_support()
        X_select_features = X.iloc[:, features]
        self.df1 = pd.concat([X_select_features, self.df_categ, self.Y], axis=1)

        return self.df1
    # End Forward feature selection methode

    # start Recursive Feature Elimination methode
    def rfe(self, type, n_features="auto"):
        X = self.df_numeric
        y = np.ravel(np.array(self.Y))
        if type == 'Regression' :
            model = LinearRegression()
        elif type == 'Classification' :
            model =  LogisticRegression()

        selector = RFE(model, n_features_to_select = n_features)
        selector.fit(X, y)
        features = selector.support_

        X.iloc[:, features]
        X_select_features = X.iloc[:, features]
        self.df1 = pd.concat([X_select_features, self.df_categ, self.Y], axis=1)

        return self.df1
    # End Recursive Feature Elimination methode

def method_chosing_feature_selc(classTest, method, param = None):
    for method in classTest.method:
        print("====method====")
        print(method)
        if(method == "mic"):
            if param != None:
                classTest.df = classTest.mic(rate = param)
            else: classTest.df = classTest.mic()
            
        if(method == "SelectK"):
            if param != None:
                classTest.df = classTest.SelectK(k = param)
            else: classTest.df = classTest.SelectK()
        if(method == "ffs"):
            if param != None:
                classTest.df = classTest.ffs(n_features = param)
            else: classTest.df = classTest.ffs()
        if(method == "rfe"):
            if param != None:
                classTest.df = classTest.rfe(n_features = param)
            else: classTest.df = classTest.rfe()
    return classTest.df1



"""
************* Feature Secaling *************
"""
# Class data Transformation (Feature Scaling)
class FeatureScaling:

    # simple constructor
    def __init__(self, df):
        self.df = df
        self.df_num = df.select_dtypes(include=np.number)
        self.num_columns = self.df_num.columns
        # dataframe only numerical
    # End unit

    # Z-score method u = x-u/s
    def z_score(self, column_name):
        col_new = np.array(self.df_num[column_name]).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(col_new)
        col_new = scaler.transform(col_new)
        return col_new
    # End Z-score

    # Min-Max method [-1,1] ~> [a,b]
    def Min_Max(self, column_name, min=0, max=1):
        col_new = np.array(self.df_num[column_name]).reshape(-1, 1)
        scaler = MinMaxScaler((min, max))
        scaler.fit(col_new)
        col_new = scaler.transform(col_new)
        return col_new

    # Max-Absolute scaling Z = X / max(absX)
    def MaxAbsoluteSc(self, column_name):
        col_new = np.array(self.df_num[column_name]).reshape(-1, 1)
        return col_new/abs(col_new).max()

    # unit length method Zi = Xi / ||X||
    def unit_length(self, column_name, norm):
        return normalize(np.array(self.df_num[column_name]).reshape(-1, 1), norm, axis=0)

    # Mean standarization Z  = x-mean/(max-min)
    def mean_std(self, column_name):
        col = np.array(self.df_num[column_name]).reshape(-1, 1)
        return (col - col.mean())/(col.max()-col.min())

    # Robust Scaling X = (x-median) / IQR
    def Robust_scaler(self, column_name):
        col_new = np.array(self.df_num[column_name]).reshape(-1, 1)
        scaler = RobustScaler()
        scaler.fit(col_new)
        col_new = scaler.transform(col_new)
        return col_new

def method_chosing_feature_selc(Fsc, method, param = None):
        # print("====method====")
        # print(method)
        if(method == "z_score"):
            if param != None:
                new_df = Fsc.df 
                new_df[param] = Fsc.z_score(column_name = param)
            else: pass #if the user doesn't enter the param, we will return the same dataframe
            
        if(method == "Min_Max"):
            if param != None:
                new_df = Fsc.df
                new_df[param] = Fsc.Min_Max(column_name = param)
            else: pass 
        if(method == "MaxAbsoluteSc"):
            if param != None:
                new_df = Fsc.df
                new_df[param] = Fsc.MaxAbsoluteSc(column_name = param)
            else: pass
        if(method == "unit_length"):
            if param != None:
                new_df = Fsc.df
                new_df[param] = Fsc.unit_length(column_name = param)
            else: Fsc.df = Fsc.rfe()
        if(method == "mean_std"):
            if param != None:
                new_df = Fsc.df
                new_df[param] = Fsc.mean_std(column_name = param)
            else: pass
        return new_df

