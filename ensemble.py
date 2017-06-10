import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, RBF, DotProduct, RationalQuadratic
from models import AverageEnsemble, StackingEnsemble
from models import evaludate_model, evaludate_submodels
from preprocess import load_data

x_train, x_test, y_train, index = load_data()

# 42
SEED = 42

regr1 = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,  # 0.01
                 max_depth=4,  # 4
                 min_child_weight=1.5,
                 n_estimators=7200,  # 3000
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 random_state=SEED,
                 silent=1)

best_alpha = 0.00099
regr2 = Lasso(alpha=best_alpha, max_iter=50000)

regr3 = ElasticNet(alpha=0.001)

regr4 = KernelRidge(alpha=0.3, kernel='polynomial', degree=2, coef0=1.85)

# regr5 = svm.SVR(kernel='rbf')

kernel = 1.0**2 * Matern(length_scale=1.0,
                         length_scale_bounds=(1e-05, 100000.0), nu=0.5)
regr5 = GaussianProcessRegressor(kernel=kernel, alpha=5e-9,
                                 optimizer='fmin_l_bfgs_b',
                                 n_restarts_optimizer=0,
                                 normalize_y=False,
                                 copy_X_train=True,
                                 random_state=SEED)

en_regr = RandomForestRegressor(n_estimators=200, max_features='auto',
                                max_depth=12, min_samples_leaf=2)

# regr6 = ExtraTreesRegressor(n_estimators=200, max_features=24,
#                            max_depth=13, min_samples_leaf=2)

# AverageEnsemble
regr = AverageEnsemble([regr1, regr3])

# StackingEnsemble
# regr = StackingEnsemble(5, en_regr, [regr1, regr2, regr3, regr4])

print('Evaluating each model separately..')
evaludate_submodels(regr, x_train, y_train)

print('Evaluating ensemble..')
evaludate_model(regr, x_train, y_train)

print('Fitting ensemble and predicting..')
# AverageEnsemble
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

# StackingEnsemble
# y_pred = regr.fit_predict(x_train, y_train, x_test)
print('Saving results..')
y_pred = np.expm1(y_pred)

pred_df = pd.DataFrame(y_pred, index=index, columns=['SalePrice'])
pred_df.to_csv('submission.csv', header=True, index_label='Id')
