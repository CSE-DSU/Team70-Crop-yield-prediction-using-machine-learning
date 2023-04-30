

# importing necessary libraries 

import numpy as np
import pandas as pd
import pickle

# loading the dataset

crop_data=pd.read_csv("crop_production.csv")

# Dropping missing values 
crop_data = crop_data.dropna()

# Displaying State Names present in the dataset
crop_data[crop_data.State_Name=='Andaman and Nicobar Islands'].District_Name.unique()

# Adding a new column Yield which indicates Production per unit Area. 

crop_data['Yield'] = (crop_data['Production'] / crop_data['Area'])
crop_data.head(10)

# Dropping unnecessary columns

data = crop_data.drop(['State_Name'], axis = 1)

dummy = pd.get_dummies(data)
# print(dummy)

"""<b><i> Splitting dataset into train and test dataset </i></b>"""

from sklearn.model_selection import train_test_split

x = dummy.drop(["Production","Yield","Crop_Year"], axis=1)
y = dummy["Production"]

print(x['Crop_Arecanut'])
# print

# # Splitting data set - 25% test dataset and 75% 

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=5)

# print("x_train :",x_train.shape)
# print("x_test :",x_test.columns)
# # print("y_train :",y_train.shape)
# # print("y_test :",y_test.shape)



# # from sklearn.ensemble import RandomForestRegressor
# # model = RandomForestRegressor(n_estimators = 11)
# # model.fit(x_train,y_train)



# filename = 'random_forest_model.pkl'
# with open(filename, 'rb') as file:
#     model = pickle.load(file)

# model.score(x_test,y_test)
# rf_predict = model.predict(x_test)
# # Calculating R2 score

# from sklearn.metrics import r2_score
# r1 = r2_score(y_test,rf_predict)
# print("R2 score : ",r1)



# # # Random Forest Regression

# # from sklearn.model_selection import RandomizedSearchCV
# # rs = RandomizedSearchCV({
# #     'c': [1,10,20],
# #     'kernel' : ['rbf','linear']
# # },
# #  cv = 5,
# #  return_train_score=False,
# #  n_iter=2
# # )
# # rs.fit(x_train,y_train)
# # pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]





