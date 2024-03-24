#!/usr/bin/env python
# coding: utf-8

# In[120]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[121]:


data  = pd.read_csv('Weather_Data.csv')


# In[122]:


data.info()


# In[123]:


data.head()
data.tail()


# In[124]:


# Convert date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])


# In[125]:


# Extract additional features from date (e.g., month, day)
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day


# In[126]:


# Drop date column as we have extracted useful features
data.drop(['Date'], axis=1, inplace=True)
data.drop(['WindDir3pm'], axis =1, inplace=True)
data.drop(['WindDir9am'], axis =1, inplace=True)
data.drop(['WindGustDir'], axis =1, inplace=True)
data.drop(['RainToday'], axis =1, inplace=True)
data.drop(['RainTomorrow'], axis =1, inplace=True)


# In[127]:


# Split features and target variable
X = data.drop('Rainfall', axis=1)
y = data['Rainfall']


# In[128]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[129]:


# Define a pipeline for preprocessing and modeling
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale features
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))  # RandomForestRegressor model
])


# In[130]:


# Train the model using the pipeline
pipeline.fit(X_train, y_train)


# In[131]:


# Predictions
y_pred = pipeline.predict(X_test)


# In[132]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)


# In[133]:


# Visualize actual vs. predicted rainfall
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
plt.title('Actual vs. Predicted Rainfall')
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
plt.grid(True)
plt.show()


# In[ ]:




