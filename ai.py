from unittest import result
import seaborn as sns
import numpy as np

#Get the data
titanic_data = sns.load_dataset('titanic')
titanic_data.head()
df = titanic_data.copy()

#Methode class
class MissingValues:
  
  #init function
  def __init__(self, dataFrame, methods, rate = 0.1):
    self.rate = rate
    self.dataFrame = dataFrame
    self.methods = methods

    self.df_numeric = dataFrame.select_dtypes(include=[np.number])
    self.numeric_cols = self.df_numeric.columns.values
    self.missing_numeric_columns = []
    for column in self.df_numeric.columns:
       if self.df_numeric[column].isnull().mean() > self.rate:
          self.missing_numeric_columns.append(column)
    
    self.df_categorical = dataFrame.select_dtypes(exclude=[np.number])
    self.categorical_cols = self.df_categorical .columns.values
    self.missing_categorical_columns = []
    for column in self.df_categorical.columns:
       if self.df_categorical[column].isnull().mean() > self.rate:
          self.missing_categorical_columns.append(column)
  #End init function


  # Row Imputation methode
  def impute_nan_row(self):
    self.df_copy = self.dataFrame.copy()
    self.df_copy = self.df_copy.dropna(axis=0)
    return self.df_copy
  #End row Imputation methode

  # Column Imputation methode
  def impute_nan_column(self):
    self.df_copy = self.dataFrame.copy()
    rate_columns = []
    for column in self.df_copy.columns:
       if self.df_copy[column].isnull().mean() > self.rate:
          self.rate_columns.append(column)

    self.df_copy = self.df_copy.drop(columns=rate_columns, axis=1)
    return self.df_copy
  #End column Imputation methode

  # Mean methode
  def impute_nan_mean(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_numeric_columns:
      mean = self.df_copy[column].mean()
      self.df_copy[column].fillna(mean, inplace=True)
    return self.df_copy
  #End Mean methode

  # median methode
  def impute_nan_median(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_numeric_columns:
      median = self.df_copy[column].median()
      self.df_copy[column].fillna(median, inplace=True)
    return self.df_copy
  #End median methode

  # End of Distribution Imputation methode
  def impute_nan_eod(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_numeric_columns:
      eod_value = self.df_copy[column].mean() + 3 * self.df_copy[column].std()
      self.df_copy[column].fillna(eod_value, inplace=True)
    return self.df_copy
  #End "End of Distribution Imputation" methode

  # Arbitrary Value Imputation methode
  def impute_nan_arbitrary_value(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_numeric_columns:
      max = self.df_copy[column].max()
      arb_val = self.arb_value(max)
      self.df_copy[column].fillna(arb_val, inplace=True)
    return self.df_copy
  #End Arbitrary Value Imputation methode

  # mode methode
  def impute_nan_mode(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_categorical_columns:
      self.df_copy[column].fillna(self.df_copy[column].mode().iloc[0], inplace=True)
    return self.df_copy
  #End mode methode

  # Arbitrary value columns methode
  def impute_nan_arbitrary_columns(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_categorical_columns:
      self.df_copy[column].fillna("Misssing_value", inplace=True)
    return self.df_copy
  #End Arbitrary value columns methode


  def arb_value(self, max):
    if(max < 10): return 9
    elif(max < 100): return 99
    elif(max < 1000): return 999
    elif(max < 10000): return 9999
    elif(max < 100000): return 99999
    elif(max < 1000000): return 999999
    elif(max < 10000000): return 9999999


def method_chosing(classTest):
    for method in classTest.methods:
      if(method == "row"): return classTest.impute_nan_row()
      elif(method == "column"): return classTest.impute_nan_column()
      elif(method == "mean"): return classTest.impute_nan_mean()
      elif(method == "median"): return classTest.impute_nan_median()
      elif(method == "eod"): return classTest.impute_nan_eod()
      elif(method == "arbitrary"): return classTest.impute_nan_arbitrary_value()
      elif(method == "mode"): return classTest.impute_nan_mode()
      elif(method == "arbitraryCat"): return classTest.impute_nan_arbitrary_columns()



#print("type: ", type(df.isnull().mean()))
#print(df.isnull().mean()[0])

results =  df.isnull().mean().to_dict()

#print("type: ", type(resultDict))

#print("dict: ", resultDict)

#for key in resultDict:
#   print(key, '->', resultDict[key])


#results = df.isnull().mean()
#data = ""
