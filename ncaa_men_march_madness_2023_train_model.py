import numpy as np
import pickle

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, log_loss

def train_model(X, y):
	param_grid = {
		'n_estimators': [100, 300],
		'max_depth': [3, 5, 7],
		'learning_rate': [0.01, 0.1],
		'subsample': [0.8],
		'colsample_bytree': [0.8],
		'min_child_weight': [1, 3],
	}

	X = np.array(X)
	y = np.array(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

	model = XGBClassifier(random_state=42, eval_metric='logloss')

	grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_log_loss', verbose=2, refit=True, n_jobs=-1)

	grid_search.fit(X_train, y_train)

	print('Best hyperparameters:', grid_search.best_params_)

	best_model = grid_search.best_estimator_

	with open("../Models/model_9.sav", "wb") as f:
		pickle.dump(best_model, f)

	y_pred = best_model.predict(X_test)
	y_pred_proba = best_model.predict_proba(X_test)[:, 1]
	accuracy = accuracy_score(y_test, y_pred)
	logloss = log_loss(y_test, y_pred_proba)
	print("Accuracy: ", accuracy)
	print("Log Loss: ", logloss)


X = pickle.load(open("../Pickles/X.pickle", "rb"))
y = pickle.load(open("../Pickles/y.pickle", "rb"))

train_model(X, y)
