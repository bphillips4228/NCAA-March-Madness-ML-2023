import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, log_loss

def train_model(X, y):
	param_grid = {
		'n_estimators': [100],
		'max_depth': [10],
		'min_samples_split': [2],
		'min_samples_leaf': [4],
		'max_features': ['sqrt']
	}

	X = np.array(X)
	y = np.array(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

	model = RandomForestRegressor(random_state=42)

	grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=2, refit=True, n_jobs=8)
	
	grid_search.fit(X,y)

	print('Best hyperparameters:', grid_search.best_params_)

	best_model = grid_search.best_estimator_


	pickle_out = open("../Models/model_9.sav", "wb")
	pickle.dump(best_model, pickle_out)
	pickle_out.close()


	y_pred = best_model.predict(X)
	mse = mean_squared_error(y, y_pred)
	logloss = log_loss(y, y_pred)
	print("Mean squared error: ", mse)
	print("Log Loss: ", logloss)


X = pickle.load(open("../Pickles/X.pickle", "rb"))
y = pickle.load(open("../Pickles/y.pickle", "rb"))

train_model(X, y)
