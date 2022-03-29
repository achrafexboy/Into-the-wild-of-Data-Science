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
  def __init__(self, dataFrame, methode, rate = 0.1):
    self.rate = rate
    self.dataFrame = dataFrame
    self.methode = methode
    self.df_numeric = dataFrame.select_dtypes(include=[np.number])
    self.numeric_cols = self.df_numeric.columns.values
    self.missing_columns = []
    for column in self.df_numeric.columns:
       if self.df_numeric[column].isnull().mean() > self.rate:
          self.missing_columns.append(column)
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
    self.df_copy = self.df_copy.dropna(axis=1)
    return self.df_copy
  #End column Imputation methode

  # Mean methode
  def impute_nan_mean(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_columns:
      mean = self.df_copy[column].mean()
      self.df_copy[column].fillna(mean, inplace=True)
    return self.df_copy
  #End Mean methode

  # median methode
  def impute_nan_median(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_columns:
      median = self.df_copy[column].median()
      self.df_copy[column].fillna(median, inplace=True)
    return self.df_copy
  #End median methode

  # End of Distribution Imputation methode
  def impute_nan_eod(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_columns:
      eod_value = self.df_copy[column].mean() + 3 * self.df_copy[column].std()
      median = self.df_copy[column].median()
      self.df_copy[column].fillna(eod_value, inplace=True)
    return self.df_copy
  #End "End of Distribution Imputation" methode

  # Arbitrary Value Imputation methode
  def impute_nan_arbitrary_value(self):
    self.df_copy = self.dataFrame.copy()
    for column in self.missing_columns:
      max = self.df_copy[column].max()
      arb_val = self.arb_value(max)
      self.df_copy[column].fillna(arb_val, inplace=True)
    return self.df_copy
  #End Arbitrary Value Imputation methode

  # Regression Model Imputation methode ##TODO##
  #def impute_nan_regression(self):
  #End Regression Model Imputation methode

  def arb_value(self, max):
    if(max < 10): return 9
    elif(max < 100): return 99
    elif(max < 1000): return 999
    elif(max < 10000): return 9999
    elif(max < 100000): return 99999
    elif(max < 1000000): return 999999
    elif(max < 10000000): return 9999999


def methode_chosing(classTest):
    if(classTest.methode == "row"): return classTest.impute_nan_row()
    elif(classTest.methode == "column"): return classTest.impute_nan_column()
    elif(classTest.methode == "mean"): return classTest.impute_nan_mean()
    elif(classTest.methode == "median"): return classTest.impute_nan_median()
    elif(classTest.methode == "eod"): return classTest.impute_nan_eod()



print("type: ", type(df.isnull().mean()))
print(df.isnull().mean()[0])

resultDict =  df.isnull().mean().to_dict()

print("type: ", type(resultDict))

print("dict: ", resultDict)

for key in resultDict:
   print(key, '->', resultDict[key])


results = df.isnull().mean()
data = ""
