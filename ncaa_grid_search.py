import itertools
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, log_loss

try:
	from xgboost import XGBRegressor
	USE_XGB = True
except ImportError:
	from sklearn.ensemble import GradientBoostingRegressor
	USE_XGB = False
	print("XGBoost not available, using sklearn GradientBoostingRegressor")

from ncaa_men_march_madness_2023_prepare_data import DataParams, build_season_data_parameterized

# --- Configuration ---
TEAM_CSV = '../data/2023/MTeams.csv'
SEASON_CSV = '../data/2023/MRegularSeasonDetailedResults.csv'
TOURNEY_CSV = '../data/2023/MNCAATourneyDetailedResults.csv'

START_YEAR = 2003
PREDICTION_YEAR = 2023

# High-impact parameters to search over.
# Reduce grid values or comment out parameters to speed up the search.
PARAM_GRID = {
	'home_win_weight':      [0.6, 0.8, 1.0],
	'away_win_weight':      [1.0, 1.2, 1.4],
	'glicko_tau':           [0.3, 0.5, 0.7],
	'lookback_n':           [3, 5, 7],
}

# Secondary parameters — uncomment to include in search (multiplies runtime)
# 'glicko_carryover_denom': [150, 200, 300],
# 'mov_cap':                [7, 10, 15],


def build_model():
	"""Create model with fixed hyperparameters for fair comparison across data params."""
	if USE_XGB:
		return XGBRegressor(
			n_estimators=300, max_depth=5, learning_rate=0.1,
			subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
			random_state=42, objective='reg:logistic'
		)
	else:
		return GradientBoostingRegressor(
			n_estimators=300, max_depth=5, learning_rate=0.1,
			subsample=0.8, min_samples_leaf=1, random_state=42
		)


def evaluate_params(params, all_data, team_names):
	"""Build features with given params, evaluate with 5-fold CV log loss."""
	X, y, _ = build_season_data_parameterized(
		all_data, team_names,
		sy=START_YEAR, py=PREDICTION_YEAR,
		params=params
	)

	model = build_model()
	scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=False)
	scores = cross_val_score(model, X, y, cv=5, scoring=scorer, n_jobs=-1)
	return -scores.mean()  # positive log_loss (lower is better)


def main():
	print("Loading data...")
	season_data = pd.read_csv(SEASON_CSV)
	tournament_data = pd.read_csv(TOURNEY_CSV)
	team_names = pd.read_csv(TEAM_CSV)

	all_data = pd.concat([season_data, tournament_data], ignore_index=True)
	all_data = all_data.sort_values(by=['Season', 'DayNum'])

	# Build all parameter combinations
	keys = list(PARAM_GRID.keys())
	values = list(PARAM_GRID.values())
	combos = list(itertools.product(*values))
	print(f"Total parameter combinations: {len(combos)}")

	results = []
	best_score = float('inf')
	best_params = None

	for i, combo in enumerate(combos):
		param_dict = dict(zip(keys, combo))
		params = DataParams(**param_dict)

		start = time.time()
		try:
			score = evaluate_params(params, all_data, team_names)
		except Exception as e:
			print(f"[{i+1}/{len(combos)}] FAILED: {e}  params={param_dict}")
			continue
		elapsed = time.time() - start

		marker = ""
		if score < best_score:
			best_score = score
			best_params = param_dict
			marker = " ** NEW BEST **"

		results.append({'params': param_dict, 'log_loss': score, 'time': round(elapsed, 1)})
		print(f"[{i+1}/{len(combos)}] log_loss={score:.6f}  time={elapsed:.1f}s  params={param_dict}{marker}")

	# Sort and report
	results.sort(key=lambda x: x['log_loss'])
	print("\n" + "=" * 60)
	print("TOP 5 PARAMETER COMBINATIONS")
	print("=" * 60)
	for rank, r in enumerate(results[:5], 1):
		print(f"  #{rank}  log_loss={r['log_loss']:.6f}  params={r['params']}")

	print(f"\nBest parameters: {best_params}")
	print(f"Best log_loss:   {best_score:.6f}")

	# Save results
	with open('../Results/grid_search_results.json', 'w') as f:
		json.dump(results, f, indent=2)
	print("Results saved to ../Results/grid_search_results.json")


if __name__ == '__main__':
	main()
