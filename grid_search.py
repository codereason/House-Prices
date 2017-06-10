import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from models import AverageEnsemble, StackingEnsemble
from models import evaludate_model, evaludate_submodels
from preprocess import load_data
from models import rmse

x_train, x_test, y_train, index = load_data()
RMSE = make_scorer(rmse)

regr = ExtraTreesRegressor(max_depth=13, min_samples_leaf=2)

param_grid = {'n_estimators': [65, 60, 70, 75, 100, 200],
              'max_features': [22, 23, 24]}
model = GridSearchCV(estimator=regr, param_grid=param_grid,
                     n_jobs=1, cv=10, verbose=20, scoring=RMSE)
model.fit(x_train, y_train)
