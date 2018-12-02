import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv('diabetes.csv')
# #print(diabetes.columns)

# # diabetes.head()
# print("dimension of diabetes data: {}".format(diabetes.shape))
# print(diabetes.groupby('Outcome').size())

# # diabetes.info()

# # X = diabetes[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
# X = diabetes[['Pregnancies','SkinThickness','BMI','DiabetesPedigreeFunction','Age']]
# y = diabetes[['Outcome']]
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(X_train, 'x_train')
# print(X_test, 'x_test')
# print(y_train, 'y_train')
# print(y_test, 'y_test')

from sklearn import linear_model as lm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix

svc = SVC(gamma='auto',probability=True)

data_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
train, test = train_test_split(data_mod, test_size=0.20)
print(data_mod.shape)
print(train.shape)
print(test.shape)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']
target = 'Outcome'
classifiers = [
  
    SVC(kernel='linear'),
  

]
classifier_names = [
    
    
    'SVM classifier with linear kernel',
    
  
]


for clf, clf_name in zip(classifiers, classifier_names):
    cv_scores = cross_val_score(clf, train[features], train[target], cv=5)
    
    print(clf_name, ' mean accuracy: ', round(cv_scores.mean()*100, 3), '% std: ', round(cv_scores.var()*100, 3),'%')


final_model_smv_lin = SVC(kernel='linear',probability=True).fit(train[features], train[target])
# final_model_gnb = gnb().fit(train[features], train[target])

y_hat_svm = final_model_smv_lin.predict(test[features])
# y_hat_gnb = final_model_gnb.predict(test[features])

print('test accuracy for SVM classifier with a linear kernel:'\
      , round(accuracy_score(test[target], y_hat_svm)*100, 2), '%')

svc = SVC(kernel='linear',probability=True).fit(train[features], train[target])

print(svc.predict([[ 0, 30, 80, 0, 22, 40, 0, 0.627]]))
print(svc.predict_proba([[ 0, 30, 80, 0, 22, 40, 0, 0.627]]))

# print('test accuracy for Gaussian naive bayes classifier:', \
#       round(accuracy_score(test[target], y_hat_gnb)*100, 2),'%')




# cv_scores = cross_val_score(SVC(kernel='linear'), train[features], train[target], cv=5)

# print('cv_score: ',cv_scores)
# print('cv_score.mean(): ',cv_scores.mean())
# print('cv_score.var(): ',cv_scores.var())

# final_model_smv_lin = SVC(kernel='linear',probability=True).fit(train[features], train[target])

# y_hat_svm = final_model_smv_lin.predict(test[features])

# # print('predict', y_hat_svm)
# print('test accuracy for SVM classifier with a linear kernel:'\
#       , round(accuracy_score(test[target], y_hat_svm)*100, 2), '%')


# y_hat_svm = final_model_smv_lin.predict(test[features])
# print('test accuracy for SVM classifier with a linear kernel:'\
#       , round(accuracy_score(test[target], y_hat_svm)*100, 2), '%')

# print(final_model_smv_lin.predict([[ 6, 148, 72, 35, 0, 36.6, 0.627, 50]]))
# print(final_model_smv_lin.predict_proba([[ 6, 148, 72, 35, 0, 36.6, 0.627, 50]]))




# svc.fit(X_train,y_train)

# print(svc.predict_proba([[ 0, 2,  28.04, 1, 21]]))


# print(svc.predict_proba([[ 6, 148, 72, 35, 0, 36.6, 0.627, 50]]))
# print(svc.predict([[ 0, 168, 74, 0, 0,  0, 0.537, 0]]))
# print(svc.predict_proba([[ 0, 168, 74, 22, 190,  43, 0.537, 50]]))




# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)
# print(X_train, 'x_train')
# print(X_test, 'x_test')
# print(y_train, 'y_train')
# print(y_test, 'y_test')

# from sklearn.svm import SVC
# svc = SVC(gamma='auto',probability=True)
# svc.fit(X_train, y_train)
# print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
# print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))


# print("Accuracy on test set: {:.2f}".format(svc.score([[ 1, 85, 66, 29, 0, 26.6, 0.351, 31]], [0])))
# print(svc.predict([[ 10, 168, 74, 0, 0, 38, 0.537, 34]]))
# print(svc.predict_proba([[ 10, 168, 74, 0, 0, 38, 0.537, 34]]))
# print(svc.predict([[ 6, 148, 72, 35, 0, 36.6, 0.627, 50]]))
# print(svc.predict_proba([[ 6, 148, 72, 35, 0, 36.6, 0.627, 50]]))

# print(svc.predict([[ 0, 118, 84, 47, 230, 45.8, 0.551, 31]]))
# print(svc.predict_proba([[ 0, 118, 84, 47, 230, 45.8, 0.551, 31]]))
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)
# svc = SVC(gamma='auto')
# svc.fit(X_train_scaled, y_train)
# print("Accuracy on training set: {:.2f}".format(svc.score(X_train_scaled, y_train)))
# print("Accuracy on test set: {:.2f}".format(svc.score(X_test_scaled, y_test)))


# print(X_train_scaled, ': x_trian_scaler')
# print(X_test_scaled, ': x_test_scaler')

# svc = SVC(C=1000, gamma='auto')
# svc.fit(X_train_scaled, y_train)
# print("Accuracy on training set: {:.3f}".format(
#     svc.score(X_train_scaled, y_train)))
# print(svc.predict(X_train_scaled))
# print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

    