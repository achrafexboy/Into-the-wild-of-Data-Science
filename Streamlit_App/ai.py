import numpy as np
import pandas as pd

# Methode class

class MissingValues:

    # init function
    def __init__(self, dataFrame, methods, rate = 0.1):
        self.rate = rate
        self.dataFrame = dataFrame
        self.methods = methods

        # Get the missing numerical columns
        self.df_numeric = dataFrame.select_dtypes(include=[np.number])
        self.numeric_cols = self.df_numeric.columns.values
        self.missing_numeric_columns = []
        for column in self.df_numeric.columns:
            if self.df_numeric[column].isnull().mean() > self.rate:
                self.missing_numeric_columns.append(column)

        # Get the missing categorical columns
        self.df_categorical = dataFrame.select_dtypes(exclude=[np.number])
        self.categorical_cols = self.df_categorical .columns.values
        self.missing_categorical_columns = []
        for column in self.df_categorical.columns:
            if self.df_categorical[column].isnull().mean() > self.rate:
                self.missing_categorical_columns.append(column)

        self.all_missing_columns = self.missing_numeric_columns + \
            self.missing_categorical_columns
    # End init function

    # Row Imputation methode

    def impute_nan_row(self):
        self.df_copy = self.dataFrame.copy()
        self.df_copy = self.df_copy.dropna(axis=0)
        return self.df_copy
    # End row Imputation methode

    # Column Imputation methode
    def impute_nan_column(self):
        self.df_copy = self.dataFrame.copy()

        self.df_copy = self.df_copy.drop(
            columns=self.all_missing_columns, axis=1)
        return self.df_copy
    # End column Imputation methode

    # Mean methode
    def impute_nan_mean(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_numeric_columns:
            mean = self.df_copy[column].mean()
            self.df_copy[column].fillna(mean, inplace=True)
        return self.df_copy
    # End Mean methode

    # median methode
    def impute_nan_median(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_numeric_columns:
            median = self.df_copy[column].median()
            self.df_copy[column].fillna(median, inplace=True)
        return self.df_copy
    # End median methode

    # End of Distribution Imputation methode
    def impute_nan_eod(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_numeric_columns:
            eod_value = self.df_copy[column].mean(
            ) + 3 * self.df_copy[column].std()
            self.df_copy[column].fillna(eod_value, inplace=True)
        return self.df_copy
    # End "End of Distribution Imputation" methode

    # Arbitrary Value Imputation methode
    def impute_nan_arbitrary_value(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_numeric_columns:
            max = self.df_copy[column].max()
            arb_val = self.arb_value(max)
            self.df_copy[column].fillna(arb_val, inplace=True)
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
    def impute_nan_arbitrary_columns(self):
        self.df_copy = self.dataFrame.copy()
        for column in self.missing_categorical_columns:
            self.df_copy[column].fillna("Misssing_value", inplace=True)
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

# results = df1.isnull().mean().to_dict()

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
    def __init__(self, dataFrame, method, n_components, target = None):
        self.df = dataFrame
        self.method = method
        self.target = target
        self.n_components = n_components
        # self.df.drop(target, axis=1)
        self.df_numeric = self.df.select_dtypes('number')
        self.df_numeric.fillna(0, inplace=True)
        self.df_categ = self.df.select_dtypes('category')
    # End init function
    # start Row PCA methode

    def pca(self):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_components)

        print("mocchkila")
        print(self.df_numeric.isnull().mean())

        Xtr = pca.fit_transform(self.df_numeric)

        # creating a list of column names
        column_values = [f'PC{(i+1)}' for i in range(Xtr.shape[1])]
        self.df1 = pd.DataFrame(data=Xtr, columns=column_values)
        self.df1 = pd.concat([self.df1, self.df_categ], axis=1)

        return self.df1
    # End row PCA methode

    # start factor analysis methode
    def fa(self):
        from sklearn.decomposition import FactorAnalysis
        transformer = FactorAnalysis(n_components=self.n_components , random_state=0)

        X_transformed = transformer.fit_transform(self.df_numeric)

        # creating a list of column names
        column_values = [
            f'Factor{(i+1)}' for i in range(X_transformed.shape[1])]

        self.df1 = pd.DataFrame(data=X_transformed, columns=column_values)
        self.df1 = pd.concat([self.df1, self.df_categ], axis=1)

        return self.df1
    # End factor analysis methode


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
