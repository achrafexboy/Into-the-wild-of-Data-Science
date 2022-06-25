from tkinter import N
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize
from sklearn.feature_selection import mutual_info_classif,chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, mean_squared_error
from warnings import warn


class missing_values :
    def __init__(self, dataFrame) :
        self.df = dataFrame

    def check_missing(self):
        """
        check the total number & percentage of missing values
        per variable of a pandas Dataframe
        """
        result = pd.concat([self.df.isnull().sum(), self.df.isnull().mean()],
                            axis=1)
        result = result.rename(index = str, columns = {0:'total missing', 1:'proportion'})
        return result

    def drop_missing(self, axis = 0):
        """
        Listwise deletion:
        excluding all cases (listwise) that have missing values
        Parameters
        ----------
        axis: drop rows(0) / columns(1), default axis = 0
        Returns
        -------
        Pandas dataframe with missing cases/columns dropped
        """    
        data_copy = self.df.copy(deep = True) # deep st to True to not effect the copy
        data_copy = data_copy.dropna(axis = axis, inplace = False)
        return data_copy
        

    def add_var_denote_NA(self, NA_col = []):
        """
        creating an additional variable indicating whether the data 
        was missing for that observation (1) or not (0).
        """
        data_copy = self.df.copy(deep=True)
        for i in NA_col:
            if data_copy[i].isnull().sum()>0:
                data_copy[i+'_is_NA'] = np.where(data_copy[i].isnull(),1,0)
            else:
                warn("Column '%s' has no missing cases" % i)
                
        return data_copy


    def impute_NA_with_arbitrary(self, impute_value, NA_col = []):
        """
        replacing NA with arbitrary values. 
        """
        data_copy = self.df.copy(deep=True)
        for i in NA_col:
            if data_copy[i].isnull().sum()>0:
                data_copy[i+'_'+str(impute_value)] = data_copy[i].fillna(impute_value)
            else:
                warn("Column %s has no missing cases" % i)
        return data_copy


    def impute_NA_with_avg(self, strategy = 'mean', NA_col = []):
        """
        replacing the NA with mean/median/most frequent values of that variable. 
        Note it should only be performed over training set and then propagated to test set.
        """
        data_copy = self.df.copy(deep = True)
        for i in NA_col:
            if data_copy[i].isnull().sum()>0:
                if strategy=='mean':
                    data_copy[i+'_impute_mean'] = data_copy[i].fillna(self.df[i].mean())
                elif strategy=='median':
                    data_copy[i+'_impute_median'] = data_copy[i].fillna(self.df[i].median())
                elif strategy=='mode': # replaces missing values of a categorical variable by the mode 
                                       # ... of non-missing cases of that variable.
                    data_copy[i+'_impute_mode'] = data_copy[i].fillna(self.df[i].mode()[0])
            else:
                warn("Column %s has no missing" % i)
        return data_copy            


    def impute_NA_with_end_of_distribution(self, NA_col = []):
        """
        replacing the NA by values that are at the far end of the distribution of that variable
        calculated by : mean + 3 * std
        """
        data_copy = self.df.copy(deep = True)
        for i in NA_col:
            if data_copy[i].isnull().sum() > 0:
                data_copy[i+'_impute_end_of_dist'] = data_copy[i].fillna(self.df[i].mean()+3*self.df[i].std())
            else:
                warn("Column %s has no missing" % i)
        return data_copy            
        

    def impute_NA_with_random(self, NA_col = [], random_state = 0, replace = False):
        """
        replacing the NA with random sampling 
        from the pool of available observations of the variable
        """
        data_copy = self.df.copy(deep = True)
        for i in NA_col:
            if data_copy[i].isnull().sum()>0:
                data_copy[i+'_random'] = data_copy[i]
                # extract the random sample to fill the na
                # 1- Get a random samples with a length equal to number of missing value in the 'i' column
                # 2 - indexing the samples to replace the nan values after
                random_sample = data_copy[i].dropna().sample(data_copy[i].isnull().sum(), random_state = random_state, replace = replace)
                random_sample.index = data_copy[data_copy[i].isnull()].index
                data_copy.loc[data_copy[i].isnull(), str(i)+'_random'] = random_sample
            else:
                warn("Column %s has no missing" % i)
        return data_copy

""" New part ~ Ouliers detection """

class outliers_detection :
    
    def __init__(self, dataFrame) :
        self.df = dataFrame


    def outlier_detect_arbitrary(self, col_name, upper_fence, lower_fence):
        """
        identify outliers based on arbitrary boundaries passed to the function.
        """
        para = (upper_fence, lower_fence)
        tmp = pd.concat([self.df[col_name] > upper_fence, self.df[col_name] < lower_fence], 
                        axis = 1)
        outlier_index = tmp.any(axis = 1)
        # index of outliers in the dataFrame
        # ... Remember to print it in the application
        print('Num of outlier detected:',outlier_index.value_counts()[1])
        print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))    
        return outlier_index, para


    def outlier_detect_IQR(self, col_name, threshold = 3):
        """
        outlier detection by Interquartile Ranges Rule, also known as Tukey's test. 
        1) calculate the 25th 75th quantile.
        2) calculate the IQR ( 75th quantile - 25th quantile )  
        Any value beyond:
            upper bound = 75th quantile + (IQR * threshold)
            lower bound = 25th quantile - (IQR * threshold)   
        are regarded as outliers. Default threshold is 3.
        """
        
        IQR = self.df[col_name].quantile(0.75) - self.df[col_name].quantile(0.25)
        Lower_fence = self.df[col_name].quantile(0.25) - (IQR * threshold)
        Upper_fence = self.df[col_name].quantile(0.75) + (IQR * threshold)
        para = (Upper_fence, Lower_fence)

        tmp = pd.concat([self.df[col_name] > Upper_fence, self.df[col_name] < Lower_fence],
                        axis = 1)
        outlier_index = tmp.any(axis = 1)
        # index of outliers in the dataFrame
        # ... Remember to print it in the application
        print('Num of outlier detected:',outlier_index.value_counts()[1])
        print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
        return outlier_index, para


    def outlier_detect_mean_std(self, col_name, z_score = 3): # 6-sigma default
        """
        outlier detection by Mean and Standard Deviation Method (z-score method)
        If a value is a certain number(called threshold) of standard deviations away 
        from the mean, that data point is identified as an outlier. 
        Default z-score is 3.
        Note that :
            * This method can fail to detect outliers because the outliers increase the standard deviation. 
            * The more extreme the outlier, the more the standard deviation is affected.
        """
        Upper_fence = self.df[col_name].mean() + z_score * self.df[col_name].std()
        Lower_fence = self.df[col_name].mean() - z_score * self.df[col_name].std()   
        para = (Upper_fence, Lower_fence)   
        tmp = pd.concat([self.df[col_name] > Upper_fence, self.df[col_name] < Lower_fence],
                        axis=1)
        outlier_index = tmp.any(axis=1)
        # index of outliers in the dataFrame
        # ... Remember to print it in the application
        print('Num of outlier detected:',outlier_index.value_counts()[1])
        print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
        return outlier_index, para


    def outlier_detect_MAD(self, col_name, threshold = 3): # The default threshold is 3 MAD.
        """
        outlier detection by Median Absolute Deviation Method (MAD)
        The median of the residuals is calculated. Then, the difference is calculated between each historical value and this median. 
        These differences are expressed as their absolute values, and a new median is calculated and multiplied by 
        an empirically derived constant to yield the median absolute deviation (MAD). 
        If a value is a certain number of MAD away from the median of the residuals, 
        that value is classified as an outlier. 
        (A.K.A) inverse of the cumulative distribution function of (3/4)  ≃ 0.67449
        """
        median = self.df[col_name].median()
        median_absolute_deviation = np.median([np.abs(y - median) for y in self.df[col_name]])
        modified_z_scores = pd.Series([0.67449 * (y - median) / median_absolute_deviation for y in self.df[col_name]])
        outlier_index = np.abs(modified_z_scores) > threshold

        # index of outliers in the dataFrame
        # ... Remember to print it in the application
        print('Num of outlier detected:',outlier_index.value_counts()[1])
        print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
        return outlier_index


    def impute_outlier_with_arbitrary(self, outlier_index, value, col = []):
        """
        impute outliers with arbitrary value
        """
        data_copy = self.df.copy(deep = True)
        for i in col:
            data_copy.loc[outlier_index, i] = value
        return data_copy
        
        
    def windsorization(self, col, para, strategy = 'both'):
        """
        top-coding & bottom coding (capping the maximum of a distribution at an arbitrarily set value, vice versa)
        """
        data_copy = self.df.copy(deep = True)  
        if strategy == 'both':
            data_copy.loc[data_copy[col] > para[0], col] = para[0]
            data_copy.loc[data_copy[col] < para[1], col] = para[1]
        elif strategy == 'top':
            data_copy.loc[data_copy[col] > para[0], col] = para[0]
        elif strategy == 'bottom':
            data_copy.loc[data_copy[col] < para[1], col] = para[1]  
        return data_copy


    def drop_outlier(self, outlier_index):
        """
        drop the cases that are outliers
        """
        data_copy = self.df[~outlier_index]
        return data_copy


    def impute_outlier_with_avg(self, col, outlier_index, strategy = 'mean'):
        """
        impute outlier with mean/median/most frequent values of that variable.
        """
        data_copy = self.df.copy(deep = True)
        if strategy == 'mean':
            data_copy.loc[outlier_index, col] = data_copy[col].mean()
        elif strategy == 'median':
            data_copy.loc[outlier_index, col] = data_copy[col].median()
        elif strategy == 'mode':
            data_copy.loc[outlier_index, col] = data_copy[col].mode()[0]   
            
        return data_copy

class feature_scaling :
    def __init__(self, dataFrame) :
        self.df = dataFrame

    def diagnostic_plots(self, variable):
        """
        function to plot a histogram and a Q-Q plot
        side by side, for a certain variable
        """
        plt.figure(figsize=(15,6))
        plt.subplot(1, 2, 1)
        self.df[variable].hist()

        plt.subplot(1, 2, 2)
        stats.probplot(self.df[variable], dist="norm", plot=pylab)
        # we're using streamlit / jinja2 so st.write(fig)
        plt.show()
        
        
    def log_transform(self, cols = []):
        """
        Logarithmic transformation
        """
        
        data_copy = self.df.copy(deep=True)
        for i in cols:
            data_copy[i+'_log'] = np.log(data_copy[i]+1)
            print('Variable ' + i +' Q-Q plot')
            self.diagnostic_plots(data_copy, str(i+'_log'))       
        return data_copy 


    def reciprocal_transform(self, cols = []):
        """
        Reciprocal transformation
        """
        data_copy = self.df.copy(deep=True)
        for i in cols:
            data_copy[i+'_reciprocal'] = 1/(data_copy[i])
            print('Variable ' + i +' Q-Q plot')
            self.diagnostic_plots(data_copy, str(i+'_reciprocal'))       
        return data_copy 


    def square_root_transform(self, cols = []):
        """
        square root transformation
        """
        data_copy = self.df.copy(deep = True)
        for i in cols:
            data_copy[i + '_square_root'] = (data_copy[i])**(0.5)
            print('Variable ' + i +' Q-Q plot')
            self.diagnostic_plots(data_copy, str(i+'_square_root'))        
        return data_copy 


    def exp_transform(self, coef, cols = []):
        """
        exp transformation
        """
        data_copy = self.df.copy(deep=True)
        for i in cols:
            data_copy[i+'_exp'] = (data_copy[i])**coef
            print('Variable ' + i +' Q-Q plot')
            self.diagnostic_plots(data_copy,str(i+'_exp'))         
        return data_copy

    def z_score_transform(self, cols = []):
        """
        Standard scaling of cols
        new_cols = [previous_cols - mean(previous_cols)]/std(previous_cols)
        """
        scaler = StandardScaler()
        data_copy = self.df.copy(deep = True)
        for i in cols:
            scaler.fit(data_copy[i])
            data_copy[i+'_stdScaler'] = scaler.transform(data_copy[i])
            print('Variable ' + i +' Q-Q plot')
            self.diagnostic_plots(data_copy, str(i+'_stdScaler')) 
        return data_copy

    def MinMax_transform(self, cols = [], min = 0, max = 1):
        """
        Transformation between a and b
        default a = 0, b = 1
        """
        scaler = MinMaxScaler((min, max))
        data_copy = self.df.copy(deep = True)
        for i in cols:
            scaler.fit(data_copy[i])
            data_copy[i+'_MinMaxScaler'] = scaler.transform(data_copy[i])
            print('Variable ' + i +' Q-Q plot')
            # self.diagnostic_plots(data_copy, str(i+'_MinMaxScaler'))  # just because I need this method in feature selection part [25/06/2022]
        return data_copy

    def MaxAbsolute_transform(self, cols = []):
        """
        Transformation between 0 and 1
        Z = X / max(absX)
        """
        data_copy = self.df.copy(deep = True)
        for i in cols:
            data_copy[i + '_MaxAbsolute'] = data_copy[i] / abs(data_copy[i]).max()
            print('Variable ' + i +' Q-Q plot')
            self.diagnostic_plots(data_copy, str(i + '_MaxAbsolute')) 
        return data_copy

    def unit_length_transform(self, cols = [], norm = 'l2'):
        """
        Transformation between 0 and 1
        Zi = Xi / ||X||
        """
        data_copy = self.df.copy(deep = True)
        for i in cols:
            data_copy[i + '_UnitLength'] = normalize(data_copy[i], norm, axis = 0)
            print('Variable ' + i +' Q-Q plot')
            self.diagnostic_plots(data_copy, str(i + '_UnitLength')) 
        return data_copy

    def mean_std_transform(self, cols = []):
        """
        Z  = x - mean(X) / (max(X) - min(X))
        """
        data_copy = self.df.copy(deep = True)
        for i in cols:
            data_copy[i + '_MeanStd'] = (data_copy[i] - data_copy[i].mean())/(data_copy[i].max() - data_copy[i].min())
            print('Variable ' + i +' Q-Q plot')
            self.diagnostic_plots(data_copy, str(i + '_MeanStd')) 
        return data_copy

    def Robust_scaler_transform(self, cols = []):
        """
        Z = ( X - median(X)) / IQR
        """
        scaler = RobustScaler()
        data_copy = self.df.copy(deep = True)
        for i in cols:
            scaler.fit(data_copy[i])
            data_copy[i+'_RobustScaler'] = scaler.transform(data_copy[i])
            print('Variable ' + i +' Q-Q plot')
            self.diagnostic_plots(data_copy, str(i+'_RobustScaler')) 
        return data_copy


 ### Feature Selection for classification [last_edite_date][20/05/2022]
class Feature_Selection :
    def __init__(self, dataFrame) :
        self.df = dataFrame
        # init function, you need to to mentionned the split and other parameters of the methods previously if it necc 
        # ... otherwise it can be declared in the method it self, without using self !
    def rf_importance(self, X_train, y_train, max_depth = 10, class_weight = None, n_estimators = 50, random_state = 0):
        
        model = RandomForestClassifier( n_estimators = n_estimators, max_depth = max_depth,
                                        random_state = random_state, class_weight = class_weight,
                                        n_jobs = -1)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feat_labels = X_train.columns

        l1,l2,l3,l4 = [],[],[],[]
        for f in range(X_train.shape[1]):
            l1.append(f+1)
            l2.append(indices[f])
            l3.append(feat_labels[indices[f]])
            l4.append(importances[indices[f]])

        feature_rank = pd.DataFrame(zip(l1,l2,l3,l4),columns=['id','indice','feature','importances']).set_index('id')
        return feature_rank


    def gbt_importance(self, X_train, y_train, max_depth = 10, n_estimators = 50, random_state = 0):
        
        model = GradientBoostingClassifier(n_estimators = n_estimators, max_depth = max_depth,
                                        random_state = random_state)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feat_labels = X_train.columns
    
        l1,l2,l3,l4 = [],[],[],[]
        for f in range(X_train.shape[1]):
            l1.append(f+1)
            l2.append(indices[f])
            l3.append(feat_labels[indices[f]])
            l4.append(importances[indices[f]])
        feature_rank = pd.DataFrame(zip(l1,l2,l3,l4),columns=['id','indice','feature','importances']).set_index('id')
        return feature_rank



    def constant_feature_detect(self, data,threshold=0.98):
        """ 
        detect features that show the same value for the 
        majority/all of the observations (constant/quasi-constant features)
        
        Parameters
        ----------
        data : pd.Dataframe
        threshold : threshold to identify the variable as constant
            
        Returns
        -------
        list of variables names
        """
        
        data_copy = data.copy(deep=True)
        quasi_constant_feature = []  # variables are found to be almost constant
        for feature in data_copy.columns:
            predominant = (data_copy[feature].value_counts() / np.float(
                        len(data_copy))).sort_values(ascending=False).values[0]
            if predominant >= threshold:
                quasi_constant_feature.append(feature)
        return quasi_constant_feature


    def corr_feature_detect(self, data, threshold = 0.8):
        """ 
        detect highly-correlated features of a Dataframe
        Parameters
        ----------
        data : pd.Dataframe
        threshold : threshold to identify the variable correlated
            
        Returns
        -------
        pairs of correlated variables
        """
        
        corrmat = data.corr()
        corrmat = corrmat.abs().unstack() # absolute value of corr coef
        corrmat = corrmat.sort_values(ascending=False)
        corrmat = corrmat[corrmat >= threshold]
        corrmat = corrmat[corrmat < 1] # remove the digonal
        corrmat = pd.DataFrame(corrmat).reset_index()
        corrmat.columns = ['feature1', 'feature2', 'corr']
    
        grouped_feature_ls = []
        # correlated_groups = []
        column_names = corrmat.columns
        correlated_groups = pd.DataFrame(columns = column_names) # it doesn't look that good in a list so we will return our corelated groups in a dataframe
        size_group = 0
        for feature in corrmat.feature1.unique():
            if feature not in grouped_feature_ls:
        
                # find all features correlated to a single feature
                correlated_block = corrmat[corrmat.feature1 == feature]
                grouped_feature_ls = grouped_feature_ls + list(
                    correlated_block.feature2.unique()) + [feature]
        
                # append the block of features to the list

                # correlated_groups.append(correlated_block)
                correlated_groups = correlated_groups.append(correlated_block)
                size_group += 1
        return correlated_groups, size_group

    ## Problem here [not solved : 25/05/2022][solved : 02/06/2022]
    def mutual_info(self, X,y,select_k=10):
        
        if select_k >= 1:
            sel_ = SelectKBest(mutual_info_classif, k = select_k).fit(X,y)
            col = X.columns[sel_.get_support()]
            
        elif 0 < select_k < 1:
            sel_ = SelectPercentile(mutual_info_classif, percentile = select_k * 100).fit(X,y)
            col = X.columns[sel_.get_support()]   
            
        else:
            raise ValueError("select_k must be a positive number")
        
        return col
        

    def chi_square_test(self, X, y, select_k=10):
    
        """
        Compute chi-squared stats between each non-negative feature and class.
        This score should be used to evaluate categorical variables in a classification task
        """
        if select_k >= 1:
            sel_ = SelectKBest(chi2, k=select_k).fit(X,y)
            col = X.columns[sel_.get_support()]
        elif 0 < select_k < 1:
            sel_ = SelectPercentile(chi2, percentile=select_k*100).fit(X,y)
            col = X.columns[sel_.get_support()]   
        else:
            raise ValueError("select_k must be a positive number")  
        
        return col
        

    def univariate_roc_auc(self, X_train,y_train,X_test,y_test,threshold):
    
        """
        First, it builds one decision tree per feature, to predict the target
        Second, it makes predictions using the decision tree and the mentioned feature
        Third, it ranks the features according to the machine learning metric (roc-auc or mse)
        It selects the highest ranked features
        ###
        note : it workes for binary Classification until NOW
        """
        roc_values = []
        for feature in X_train.columns:
            clf = DecisionTreeClassifier()
            clf.fit(X_train[feature].to_frame(), y_train)
            y_scored = clf.predict_proba(X_test[feature].to_frame())
            roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
        roc_values = pd.Series(roc_values)
        roc_values.index = X_train.columns
        # print(roc_values.sort_values(ascending=False))
        # print(len(roc_values[roc_values > threshold]),'out of the %s featues are kept'% len(X_train.columns))
        keep_col = roc_values[roc_values > threshold]
        return keep_col
            
            
    def univariate_mse(self, X_train,y_train,X_test,y_test,threshold):
    
        """
        First, it builds one decision tree per feature, to predict the target
        Second, it makes predictions using the decision tree and the mentioned feature
        Third, it ranks the features according to the machine learning metric (roc-auc or mse)
        It selects the highest ranked features
        mse : Regression
        """
        mse_values = []
        for feature in X_train.columns:
            clf = DecisionTreeRegressor()
            clf.fit(X_train[feature].to_frame(), y_train)
            y_scored = clf.predict(X_test[feature].to_frame())
            mse_values.append(mean_squared_error(y_test, y_scored))
        mse_values = pd.Series(mse_values)
        mse_values.index = X_train.columns
        print(mse_values.sort_values(ascending=False))
        print(len(mse_values[mse_values > threshold]),'out of the %s featues are kept'% len(X_train.columns))
        keep_col = mse_values[mse_values > threshold]
        return keep_col 

class data_reduction :
    
    def __init__(self, dataFrame, name_target) :
        self.df = dataFrame
        self.X = self.df.drop(name_target, axis = 1)
        self.y = self.df[name_target]
        
    def red_LDA(self, n_compenents = None, solver = 'svd') :
        lda = LinearDiscriminantAnalysis(n_components = n_compenents, solver = solver)
        X_reduced = lda.fit_transform(self.X, self.y)
        return X_reduced

    def red_PCA(self, n_compenents = None, svd_solver = 'auto'):
        pca = PCA(n_components = n_compenents, svd_solver = svd_solver)
        X_reduced = pca.fit_transform(self.X)
        return X_reduced