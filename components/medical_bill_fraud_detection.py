# -*- coding: utf-8 -*-
"""Medical Bill Fraud Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cf5GXeE-VakANMA4zx6dvZRSUdC7eZEY
"""
import pandas as pd

import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder import AutoEncoder

#filter warnings
import warnings
"""""
def ML():
  warnings.filterwarnings("ignore")

  pd.set_option("display.max_rows",None)
  pd.set_option("display.max_columns",None)

  data = pd.read_csv("components/Healthcare Providers.csv")

  data = preprocessing(data)

  ae = AutoEncoder(hidden_neurons =[15, 10, 6, 2, 2, 6, 10, 15], epochs = 26, contamination = .002, verbose = 1)
  ae.fit(data)

  return ae
"""""

def preprocessing(data):

  # remove unnecessary columns
  drop_cols = ['index', 'National Provider Identifier', 'Last Name/Organization Name of the Provider',
      'First Name of the Provider', 'Middle Initial of the Provider','Street Address 1 of the Provider',
      'Street Address 2 of the Provider','Zip Code of the Provider',"HCPCS Code"]

  data = data.drop(drop_cols, axis = 1)

  # clean the data, make it look nicer
  cleanse = ["Average Medicare Allowed Amount", "Average Submitted Charge Amount", "Average Medicare Payment Amount", "Average Medicare Standardized Amount"]

  for col in cleanse:
    data[col] = pd.to_numeric(data[col].apply(lambda x : removeComma(str(x))), errors="ignore")

  missing_cols = ['Credentials of the Provider', 'Gender of the Provider']
  # replaces null or missing values with the first mode found
  for col in missing_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

  # binary encoding
  # binary value of a categorical datatype is split into columns
  be_cols = [var for var in data.columns if data[var].dtype == "O"]

  # removes the original column and instead adds the binary encodded columns
  for col in be_cols:
    encoder = ce.BinaryEncoder(cols = [col])
    databin = encoder.fit_transform(data[col])
    data = pd.concat([data, databin], axis = 1)
    del data[col]

  # standardization
  data_cols = data.columns
  std = StandardScaler()
  data = std.fit_transform(data)
  data = pd.DataFrame(data, columns = data_cols)

  return data

def removeComma(x):
  return x.replace(",", "")

 

