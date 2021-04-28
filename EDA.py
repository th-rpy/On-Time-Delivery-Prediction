# Import Libraries

import pandas as pd
import seaborn as sns
import numpy as np
import math
import dataframe_image as dfi
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

''' Pre-processing Step '''
# 1. Loading Dataset

data = pd.read_csv('DataSet\deliveryDataset.csv', sep=',')
print(data.head(5))  # View the first 5 rows
print(data.info())  # Display the features's types
dfi.export(data.head(5), "Outputs/data.png")


"""type(Reached_YN) = Integer : Puisque notre probleme est de classification (binaire) donc il faut convertir la variable cible (Reached_On_Time)
de Integer to Object. Ainsi, la variable ID n'a aucune importance, nous allons donc la supprimer."""

data.rename(columns={'Reached.on.Time_Y.N': 'Reached_YN'},
            inplace=True)  # Rename the Label features
data['Reached_YN'] = data['Reached_YN'].astype(np.object)
data.drop('ID', axis=1, inplace=True)

# 2. Missing Values in Dataset: Check if there are a NaN value

all_NaN_values = data.isnull().sum().sort_values(
    ascending=False)  # counts NaN values by each features
# Percentage of NaN value (for each features)
percent_NaN = (data.isnull().sum()*100 / data.isnull().count()
               ).sort_values(ascending=False)
NaN_df = pd.concat([all_NaN_values, percent_NaN], axis=1, keys=[
                   'Total NaN', 'Percent NaN'])  # return the NaN dataframe

""" Nous voyons qu'il n'y a pas de valeurs invalides, nous pouvons donc poursuivre l'exploration."""
print(NaN_df)
dfi.export(NaN_df, "Outputs/NaNDf_percent.png")

# Drop the Outliers
cont_var = list(data.select_dtypes(exclude=['object']).columns)
clf = LocalOutlierFactor(n_neighbors=3)
outliers = clf.fit_predict(data[cont_var])

unique, counts = np.unique(outliers, return_counts=True)
dict(zip(unique, counts))
data['outliers']=outliers
outliers_index=list(data[data['outliers']==-1].index)
# Remove the outliers
for o in outliers_index:
      data.drop(index=o,inplace=True)

data.drop('outliers', axis = 1, inplace = True)

# 3. Describe the Features
print(data.describe(include=np.number))  # Continuous variables
print(data.describe(include=np.object))  # Categorical variables
dfi.export(data.describe(include=np.object), "Outputs/Catego_Vars_Desc.png")
dfi.export(data.describe(include=np.number), "Outputs/Continu_Vars_Desc.png")

"""
A partir de l'exploration initiale ci-dessus, nous voyons que :

- Environ 68%. des modes d'expédition sont par bateau, les autres étant le vol et la route.
- Environ 48%. de l'importance du produit est classée comme faible.
- Le sexe des clients semble être réparti de manière égale avec environ 50,4%. de femmes.
- Environ 60%. des colis ne sont pas livré en temps (On peut dire qu'on une Balanced dataset)
"""

# 4. Data Visualizations
catg_var = list(data.select_dtypes(include=['object']).columns)
cont_var = list(data.select_dtypes(exclude=['object']).columns)

sns.set_theme(style="darkgrid")

# Pie chart : Warehouse Block
plt.pie(data.Warehouse_block.value_counts(), explode=[.8, .3, .2, .1, .1], startangle=90, autopct='%.2f%%', labels=[
        'F', 'D', 'A', 'B', 'C'], radius=10, colors=['blue', 'pink', 'red', 'yellow', 'green'])
plt.axis('equal')
plt.title('Warehouse Block', fontdict={'fontsize': 22, 'fontweight': 'bold'})
plt.savefig('Outputs/PieChart/Pie_WarehouseBlock.png')
plt.show()

# Pie chart : Mode of Shipment
plt.pie(data.Mode_of_Shipment.value_counts(), explode=[.8, .3, .2], startangle=90, autopct='%.2f%%', labels=[
        'Ship', 'Flight', 'Road'], radius=10, colors=['blue', 'red', 'green'])
plt.axis('equal')
plt.title('Mode of Shipment', fontdict={'fontsize': 22, 'fontweight': 'bold'})
plt.savefig('Outputs/PieChart/Pie_Mode_of_Shipment.png')
plt.show()
# Visualizations for the categories features
for i, var in enumerate(catg_var[:-1]):
    plt.figure(i)
    sns.countplot(data=data, x=var, hue="Reached_YN").figure.savefig(
        "Outputs/CountPlot/cnt_plot_ReachedYN_{}.png".format(var))

# Visualizations for the continuous features
for i, var_ in enumerate(cont_var):
    plt.figure(i)
    sns.boxplot(data=data, y=var_, x="Reached_YN").figure.savefig(
        "Outputs/BoxPlot/cnt_plot_ReachedYN_{}.png".format(var_))


# Correlation between Reached_YN and all variables

Warehouse_block_dummies = pd.get_dummies(data['Warehouse_block'])
Mode_of_Shipment_dummies = pd.get_dummies(data['Mode_of_Shipment'])
Product_importance_dummies = pd.get_dummies(data['Product_importance'])
Gender_dummies = pd.get_dummies(data['Gender'])
Reached_on_Time_Y_N_dummies = pd.get_dummies(data['Reached_YN'])

data_corr = pd.concat([data, Warehouse_block_dummies, Mode_of_Shipment_dummies,
                      Product_importance_dummies, Gender_dummies, Reached_on_Time_Y_N_dummies], axis=1)

plt.figure(figsize=(16, 16), dpi=100)
sns.heatmap(data.corr())
ax_ = sns.heatmap(data_corr.corr(), annot=True, vmin=-1)
ax_.figure.savefig('Outputs/CorrelationHeadmap.png')
