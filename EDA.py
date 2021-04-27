# Import Libraries

import pandas as pd 
import seaborn as sns 
import numpy  as np
import math

''' Pre-processing Step '''
'''1. Loading Dataset'''

data = pd.read_csv('DataSet\deliveryDataset.csv', sep = ',')
print(data.head(5)) #View the first 5 rows 
print(data.info()) #Display the features's types



