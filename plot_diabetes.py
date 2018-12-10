import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = pd.read_csv('diabetes.csv') # importing our data into the notebook
print(diabetes.head(5)) # Examine first columns
print("") # spacing for better reading
print(diabetes.columns)

print("")
print(diabetes.describe())


data = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']

print("")
print(data.describe())
data[features].hist()
plt.show()

# plt.figure(figsize=(10,3))
# bp_pivot = data.groupby('BloodPressure').Outcome.count().reset_index()
# sns.distplot(data[data.Outcome == 0]['BloodPressure'], color='blue', label='0 Class')
# sns.distplot(data[data.Outcome == 1]['BloodPressure'], color='red', label='1 Class')
# plt.legend()
# plt.title('Number of people with blood pressure values')

# plt.figure(figsize=(10,3))
# glucose_pivot = data.groupby('Glucose').Outcome.count().reset_index()
# sns.distplot(data[data.Outcome == 0]['Glucose'], color='blue', label='0 Class')
# sns.distplot(data[data.Outcome == 1]['Glucose'], color='red', label='1 class')
# plt.legend()
# plt.title('Number of people with Glucose values')

# plt.figure(figsize=(10,3))
# BMI_pivot = data.groupby('BMI').Outcome.count().reset_index()
# sns.distplot(data[data.Outcome == 0]['BMI'], color='blue', label='Class 0')
# sns.distplot(data[data.Outcome == 1]['BMI'], color='red', label='Class 1')
# plt.legend()
# plt.title('Number of people with BMI values')

# plt.figure(figsize=(10,3))
# Insulin_pivot = data.groupby('Insulin').Outcome.count().reset_index()
# sns.distplot(data[data.Outcome == 0]['Insulin'], color='blue', label='Class 0')
# sns.distplot(data[data.Outcome == 1]['Insulin'], color='red', label='Class 1')
# plt.legend()
# plt.title('Number of people with Insulin values')

# plt.figure(figsize=(10,3))
# SkinThickness_pivot = data.groupby('SkinThickness').Outcome.count().reset_index()
# sns.distplot(data[data.Outcome == 0]['SkinThickness'], color='blue', label='Class 0')
# sns.distplot(data[data.Outcome == 1]['SkinThickness'], color='red', label='Class 1')
# plt.legend()
# plt.title('Number of people with Skin thickness values')

# plt.show()