import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn import linear_model as lm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix

diabetes = pd.read_csv('diabetes.csv')

data_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
train, test = train_test_split(data_mod, test_size=0.2)

# X = diabetes[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
# y = diabetes[['Outcome']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)

print(data_mod.shape)
print(train.shape)
print(test.shape)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']
target = 'Outcome'
classifiers = [

    knnc(),
    dtc(),
    SVC(gamma='auto'),
    SVC(kernel='linear'),
    gnb()
  

]
classifier_names = [
    'K nearest neighbors',
    'Decision Tree Classifier',
    'SVM classifier with RBF kernel',
    'SVM classifier with linear kernel',
    'Gaussian Naive Bayes'
]


for clf, clf_name in zip(classifiers, classifier_names):
    cv_scores = cross_val_score(clf, train[features], train[target], cv=5)
    
    print(clf_name, ' mean accuracy: ', round(cv_scores.mean()*100, 3), '% std: ', round(cv_scores.var()*100, 3),'%')
    

final_model_smv_lin = SVC(kernel='linear',probability=True).fit(train[features], train[target])
# final_model_gnb = gnb().fit(train[features], train[target])

y_hat_svm = final_model_smv_lin.predict(test[features])
# y_hat_gnb = final_model_gnb.predict(test[features])

print('test accuracy for SVM classifier with a linear kernel:', round(accuracy_score(test[target], y_hat_svm)*100, 2), '%')

# svc = SVC(kernel='linear',probability=True).fit(train[features], train[target])



# --------------------- best parameter --------------------
# import the necessary modules
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# I'll add a standard scaler since SVC works better if the data is scaled.
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC(kernel = "linear"))])

# Next we'll tune hyperparameters of the estimators separately in the pipeline
param_grid = [
    {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
    'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    ]

# Train-test split,instantiate,fit and predict paradigm
# create train and test sets
train, test = train_test_split(data_mod, random_state = 2)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)



# grid search with cross-validation
grid = GridSearchCV(pipe, param_grid, cv = 5)
grid.fit(train[features], train[target])

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Test-set score: {:.2f}".format(grid.score(test[features], test[target])))
# print(train.shape)
# print(test.shape)



# -------------- Create Model with best parameter --------------
# Setting up the pipeline

steps = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
         ('SVM', SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False))]

pipeline = Pipeline(steps)

# Specifying the hyperparameter space
parameters = {'SVM__C':[100] ,
             'SVM__gamma':[0.001]}

# Create train and test sets
train, test = train_test_split(data_mod, test_size = 0.20, random_state = 2)

# Instantiate the GridSearchCV
grid2 = GridSearchCV(pipeline,parameters)

# Fit to the training set
grid2.fit(train[features], train[target])

# Predict the labels of the test set
y_pred = grid2.predict(test[features])

# Compute and print metrics

print("Accuracy: {}".format(grid2.score(test[features], test[target])))
print(classification_report(test[target], y_pred))
print("Tuned Model Parameters: {}".format(grid2.best_params_))



# predict input ตรงนี้เด้ออออ 
pre = grid2.predict([[ 0, 30, 80, 0, 22, 40, 0, 0.627]])
pre_prob = grid2.predict_proba([[ 0, 30, 80, 0, 22, 40, 0, 0.627]])
print ("")
print("Predict input: ", pre)
print("Predict Probability input: ", pre_prob)






