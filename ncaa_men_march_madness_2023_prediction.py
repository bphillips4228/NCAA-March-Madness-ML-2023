import pandas as pd
import math
import csv
import os
import numpy as np
import pickle

from ncaa_men_march_madness_2023_prepare_data import DataParams, DEFAULT_PARAMS

team_stats = pickle.load(open("../Pickles/team_stats.pickle", "rb"))
team_seeds = {}
final_data = []
prediction_year = 2023
prediction_range = [2023]

# Load best params from grid search if available, otherwise use defaults
params_path = "../Pickles/best_params.pickle"
if os.path.exists(params_path):
	params = pickle.load(open(params_path, "rb"))
	print(f"Loaded params: lookback_n={params.lookback_n}")
else:
	params = DEFAULT_PARAMS

def init_data():
	for i in range(prediction_range[0], prediction_year+1):
		team_seeds[i] = {}

def get_seed(team, season):
	try:
		return int(team_seeds[season][team][1:])
	except:
		return 16

def prepare_data(features):
	features = np.array(features)
	features = features[np.newaxis, :]
	return features

def predict_winner(team_1, team_2, model, season):
	team_1_features = team_1.get_features(params.lookback_n)
	team_2_features = team_2.get_features(params.lookback_n)

	matchup_features = [a - b for a, b in zip(team_1_features, team_2_features)]

	prepared_data = prepare_data(matchup_features)

	prediction = model.predict(prepared_data)
	return np.clip(prediction, 0, 1)

def get_teams(team_list, year):
	for i in range(len(team_list)):
			for j in range(i + 1, len(team_list)):
				if team_list[i] < team_list[j]:
					prediction = predict_winner(team_list[i], team_list[j], model, year)
					label = str(year) + '_' + str(team_list[i]) + '_' + str(team_list[j])
					final_data.append([label, prediction[0]])

init_data()

model = pickle.load(open("../Models/model.sav", "rb"))

print("Getting teams")
print("Predicting matchups")

seeds = pd.read_csv('../Data/2023/MNCAATourneySeeds.csv')
tourney_teams = []

for year in prediction_range:
	for index, row in seeds.iterrows():
		if row['Season'] == year:
			team_seeds[year][row['TeamID']] = row['Seed']
			tourney_teams.append(row['TeamID'])
	tourney_teams.sort()

	get_teams(tourney_teams, year)
	tourney_teams.clear()


print(f"Writing {len(final_data)} results")
with open('../Predictions/prediction_1.csv', 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(['ID', 'Pred'])
	writer.writerows(final_data)