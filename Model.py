# Import libraries and Data from EDA.py
from EDA import data, data_corr, catg_var, cont_var
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt 
import dataframe_image as dfi


# Split the Cleaned Dataset (data)
# Our label : Reached_YN (0,1) = (Yes, No)
label_class = data['Reached_YN'].astype('int')
data_ = data.drop('Reached_YN', axis=1)
# Before, we need to encoder the categorical variables
data['Product_importance'] = data['Product_importance'].map(
    {'low': 1, 'medium': 2, 'high': 3})
# Encoder model
encoder = LabelEncoder()
for col in catg_var[:-1]:
    data_[col] = encoder.fit_transform(data_[col])


# Split the dataset with test_size = 20% and 80% of dataset for train the model
train_features, test_features, train_label, test_label = train_test_split(
    data_, label_class, test_size=.2, random_state=42, stratify = None )

# 1. Decision Tree Model

# Define the model
Tree = DecisionTreeClassifier()
# Use the GrideSearch for tuning the hyperparameter like max_depth and the choose between 'gini' an 'entropy' criterion
modelTreeTuned = GridSearchCV(Tree, param_grid={'max_depth': range(
    4, 13), 'criterion': ['gini', 'entropy']})
# Train the Decision Tree model (the best : the GridSearch's output)
modelTreeTuned.fit(train_features, train_label)
best_params = modelTreeTuned.best_params_
print(best_params)

# Predictions
train_pred = modelTreeTuned.predict(train_features)
test_pred = modelTreeTuned.predict(test_features)

# Matrix Confusion for the Decision Tree Model
print('Classification Report of train_data \n',
      classification_report(train_label, train_pred))
print('Classification Report of test_data \n',
      classification_report(test_label, test_pred))

# Importance of the features
# Get importance from the model (model with {'criterion': 'entropy', 'max_depth': 5})
modelTree = DecisionTreeClassifier(criterion = best_params['criterion'], max_depth = best_params['max_depth'])
modelTree.fit(train_features, train_label)
importance = modelTree.feature_importances_
# Summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))

# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.savefig('DecisionTree/Importance.png')

# Save the tree in image format 
fig = plt.figure(figsize=(50,50))
_ = tree.plot_tree(modelTree, 
                   feature_names=data.columns[:-1],  
                   class_names=data.columns[-1],
                   filled=True, max_depth = 2)

fig.savefig("DecisionTree/decistion_tree.png")

# 2. Model 2 : Logistic regression

## Define the model
modelRgLog = LogisticRegression(solver='liblinear', random_state=0)

## Fitting the model with the train dataset
modelRgLog.fit(train_features, train_label)

## Make Predictions model
y_pred_rg = modelRgLog.predict(test_features)

## Classification Report model
classification_report(test_label , y_pred_rg)
