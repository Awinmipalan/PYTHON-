#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
train_data = pd.read_csv("C:\\Users\\DELL\\Downloads\\train.csv")
test_data = pd.read_csv("C:\\Users\\DELL\\Downloads\\test.csv")

# Data preprocessing
def preprocess_data(data):
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1) # Drop irrelevant columns
    data['Age'].fillna(data['Age'].median(), inplace=True) # Fill missing values in Age column
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True) # Fill missing values in Embarked column
    data['Fare'].fillna(data['Fare'].median(), inplace=True) # Fill missing values in Fare column
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}) # Map categorical values to numerical
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Splitting data into features and target variable
X = train_data.drop(['Survived', 'PassengerId'], axis=1)
y = train_data['Survived']

# Splitting the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(7,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Accuracy:", accuracy)


# In[2]:


model.summary()


# In[7]:


test_datas=test_data.drop(['PassengerId'],axis=1)


# In[8]:


test_datas


# In[4]:


test_data


# In[15]:


predictions=model.predict(test_datas)
predictions


# In[18]:


prediction_df=pd.DataFrame(predictions,columns=['predicted'])


# In[19]:


prediction_df


# In[21]:


test_data['prediction']=prediction_df


# In[22]:


test_data


# In[23]:


prdedicting=test_data[['PassengerId','prediction']]


# In[25]:


prdedicting


# In[27]:


csv_file_path='prediction.csv'
prdedicting.to_csv(csv_file_path,index=False)
print("prediction saved to:",csv_file_path)


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'data' is your dataset with three features and 'predictions' is your array of predictions
# 'data' should have columns for the three features and the prediction results
# For example:
# data = pd.DataFrame({'feature1': feature1_values, 'feature2': feature2_values, 'feature3': feature3_values})

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract features
f1 = test_data['PassengerId']
f2 = test_data['Age']
f3 = test_data['Fare']

# Plot the data points
ax.scatter(f1, f2, f3, c=predictions)

# Set labels and title
ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('F3')
ax.set_title('3D Scatter Plot of Predictions')

# Show the plot
plt.show()


# In[43]:


test_data['prediction']


# In[45]:


import numpy as np
threshold=0.5
binary_prediction=(test_data['prediction']>=threshold).astype(int)
binary_prediction


# In[47]:


test_data['binary_prediction']=binary_prediction


# In[48]:


binary_prediction


# In[54]:


percent=test_data['binary_prediction'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(percent.values,labels=percent.index,autopct='%.1f%%',startangle=90,explode=[0.1,0],colors=['green','blue'])
plt.legend(['True','false'],loc='upper right')
plt.title('survived',fontsize=14)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




