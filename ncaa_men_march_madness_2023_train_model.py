import numpy as np
import pickle

try:
	from xgboost import XGBRegressor
	USE_XGB = True
except ImportError:
	from sklearn.ensemble import GradientBoostingRegressor
	USE_XGB = False
	print("XGBoost not available, using sklearn GradientBoostingRegressor")

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, log_loss

def train_model(X, y):
	if USE_XGB:
		param_grid = {
			'n_estimators': [100, 300],
			'max_depth': [3, 5, 7],
			'learning_rate': [0.01, 0.1],
			'subsample': [0.8],
			'colsample_bytree': [0.8],
			'min_child_weight': [1, 3],
		}
		model = XGBRegressor(random_state=42, objective='reg:logistic')
	else:
		param_grid = {
			'n_estimators': [100, 300],
			'max_depth': [3, 5, 7],
			'learning_rate': [0.01, 0.1],
			'subsample': [0.8],
			'min_samples_leaf': [1, 3],
		}
		model = GradientBoostingRegressor(random_state=42, loss='squared_error')

	X = np.array(X)
	y = np.array(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

	grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, refit=True, n_jobs=-1)

	grid_search.fit(X_train, y_train)

	print('Best hyperparameters:', grid_search.best_params_)

	best_model = grid_search.best_estimator_

	with open("../Models/model_9.sav", "wb") as f:
		pickle.dump(best_model, f)

	y_pred = best_model.predict(X_test)
	y_pred_clipped = np.clip(y_pred, 0, 1)
	mse = mean_squared_error(y_test, y_pred_clipped)
	logloss = log_loss(y_test, y_pred_clipped)
	print("Mean Squared Error: ", mse)
	print("Log Loss: ", logloss)


X = pickle.load(open("../Pickles/X.pickle", "rb"))
y = pickle.load(open("../Pickles/y.pickle", "rb"))

train_model(X, y)
