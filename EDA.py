# Import Libraries

import pandas as pd 
import seaborn as sns 
import numpy  as np
import math

''' Pre-processing Step '''
# 1. Loading Dataset 

data = pd.read_csv('DataSet\deliveryDataset.csv', sep = ',')
print(data.head(5)) #View the first 5 rows 
print(data.info()) #Display the features's types

# 2. Missing Values in Dataset: Check if there are a NaN value 

all_NaN_values = data.isnull().sum().sort_values(ascending=False) #counts NaN values by each features
percent_NaN = (data.isnull().sum()*100 / data.isnull().count()).sort_values(ascending=False) #Percentage of NaN value (for each features)
NaN_df = pd.concat([all_NaN_values, percent_NaN], axis=1, keys=['Total', 'Percent']) # return the NaN dataframe

""" N'a aucune donnée manquante dans notre base de données """
print(NaN_df) 

# 3. Describe the Features 
print(data.describe(include=np.number)) #Continuous variables
print(data.describe(include=np.object)) #Categorical variables

"""
A partir de l'exploration initiale ci-dessus, nous voyons que :

- Environ 68%. des modes d'expédition sont par bateau, les autres étant le vol et la route.
- Environ 48%. de l'importance du produit est classée comme faible.
- Le sexe des clients semble être réparti de manière égale avec environ 50,4%. de femmes.
"""

# 4. 




