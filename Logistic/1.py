import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

### Create the dataset
X,y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

### Show DataFrame for X, y
print(pd.DataFrame(X))
print(pd.DataFrame(y))

### Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

### Model Training
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()

### Fitting
logistic.fit(X_train, y_train)

### Predict
y_pred = logistic.predict(X_test)
print(y_pred)

### Probability predict
y_pred_prob = logistic.predict_proba(X_test)
print(y_pred_prob)

### Performance Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
score = accuracy_score(y_test, y_pred)
print(score)
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)

# Hyperparameter Turning And Cross Validation
model = LogisticRegression()
penalty = ['l1', 'l2', 'elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]
solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

### Parameter: tham số
params = dict(penalty=penalty, C=c_values, solver=solver)

# GridSearchCV
### StratifiedKFold
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold()
## GridSearchCV
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=cv, n_jobs=-1)
print(grid)
### Fit model for grid
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)

y_pred = grid.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)



