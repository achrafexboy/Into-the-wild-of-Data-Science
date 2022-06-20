# import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use('seaborn-colorblind')

class viz :
    def __init__(self, dataFrame) :
        self.df = dataFrame

    def get_dtypes(self, drop_col = []):
        """
        the dtypes for each column of a pandas Dataframe
        """

        name_of_col = list(self.df.columns)
        num_var_list = []
        str_var_list = []
        all_var_list = []

        str_var_list = name_of_col.copy()
        for var in name_of_col:
            # check if column belongs to numeric type
            if (self.df[var].dtypes in (np.int, np.int64, np.uint, np.int32, np.float,
                                        np.float64, np.float32, np.double)):
                str_var_list.remove(var)
                num_var_list.append(var)
        # drop the omit column from list
        for var in drop_col:
            if var in str_var_list:
                str_var_list.remove(var)
            if var in num_var_list:
                num_var_list.remove(var)

        all_var_list.extend(str_var_list)
        all_var_list.extend(num_var_list)
        return str_var_list, num_var_list, all_var_list


    def describe(self ,col_name = None):
        """
        output the general description of a  pandas Dataframe
        
        """
        if col_name != None :
            return self.df[col_name].describe(include = 'all')
        else : return  self.df.describe(include = 'all')
        
        
    def discrete_var_barplot(self, col1_name, col2_name):
        """
        draw the bar-plot of a discrete variable x against y (target variable). 
        By default the bar shows the mean value of y.
        """
        plt.figure(figsize=(15,10))
        sns.barplot(x = self.df[col1_name], y = self.df[col2_name], data = self.df)
        
        
    def discrete_var_countplot(self, col_name):
        """
        draw the countplot of a discrete variable x.
        """    
        plt.figure(figsize=(15,10))
        sns.countplot(x = col_name, data = self.df)


    def discrete_var_boxplot(self, col1_name, col2_name):
        """
        draw the boxplot of a discrete variable x against y.
        """     
        plt.figure(figsize=(15,10))
        sns.boxplot(x = col1_name, y = col2_name, data=self.df)


    def continuous_var_distplot(self, col_name, bins = None):
        """
        draw the distplot of a continuous variable x.
        """    
        plt.figure(figsize=(15,10))
        sns.distplot(a = col_name, kde = False, bins = bins)  


    def scatter_plot(self, col1_name, col2_name):
        """
        draw the scatter-plot of two variables.
        """    
        plt.figure(figsize=(15,10))
        sns.scatterplot(x = col1_name, y = col2_name, data = self.df)

        
    def correlation_plot(self):
        """
        draw the correlation plot between variables.
        """    
        corrmat = self.df.corr()
        fig, ax = plt.subplots()
        fig.set_size_inches(11,11)
        sns.heatmap(corrmat, cmap = "YlGnBu", linewidths = .5, annot = True) 
        
        
    def heatmap(self, fmt = 'd'):
        """
        draw the heatmap between 2 variables.
        """    
        fig, ax = plt.subplots()
        fig.set_size_inches(11,11)
        sns.heatmap(self.df, cmap = "YlGnBu", linewidths = .5, annot = True, fmt = fmt)  