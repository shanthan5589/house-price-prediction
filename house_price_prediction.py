"""
House Price Prediction Model
Kaggle Competition: House Prices - Advanced Regression Techniques
Score: 14765.10633 (Rank: 281)
Author: Shanthan Yellapragada

"""

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor

# Load and prepare data
Data = pd.read_csv('house_price_train.csv',index_col='Id')

Data.dropna(subset=['SalePrice'],axis=0,inplace=True)

y = Data['SalePrice']

Data.drop('SalePrice',axis=1,inplace=True)

X = Data

# Identify column types
numerical_columns = [col for col in X.columns if X[col].dtype != 'object']

columns_cardinality_more_than_ten = [col for col in X.columns if X[col].dtype == 'object' and len(set(X[col])) > 10]
columns_cardinality_less_than_ten = [col for col in X.columns if X[col].dtype == 'object' and len(set(X[col])) <= 10]

# Define preprocessing pipelines
numerical_transformer = Pipeline([
    ('numeric',SimpleImputer(strategy='constant',fill_value=0))
])

more_than_ten_transformer = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
])

less_than_ten_transformer = Pipeline([
    ('OneHot',OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ('numerical',numerical_transformer,numerical_columns),
    ('more_than_ten',more_than_ten_transformer,columns_cardinality_more_than_ten),
    ('less_than_ten',less_than_ten_transformer,columns_cardinality_less_than_ten),
])

# Define model
model = XGBRegressor(n_estimators=5400,learning_rate=0.01,device='cuda')

# Create final pipeline
final_pipeline = Pipeline([
    ('preprocess',preprocess),
    ('model',model)
])

# Cross-validation
loss = cross_val_score(final_pipeline,X,y,cv=4,scoring='neg_mean_absolute_error')

print(f"Loss: {-1 * loss.mean()}")

# Load test data and make predictions
X_test = pd.read_csv('house_price_test.csv')

final_pipeline.fit(X,y)

predictions = final_pipeline.predict(X_test)

output = pd.DataFrame({
    'Id':X_test['Id'],
    'SalePrice':predictions
})

# Save submission
output.to_csv('submission.csv', index=False)