import pandas as pd
import math
import csv
import numpy as np
import pickle

team_stats = pickle.load(open("../Pickles/team_stats.pickle", "rb"))
team_seeds = {}
final_data = []
prediction_year = 2023
prediction_range = [2023]

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
	features = features[newaxis, :]
	return features

def predict_winner(team_1, team_2, model, season):
	team_1_features = team_1.get_features(5)
	team_2_features = team_2.get_features(5)

	matchup_features = [a - b for a, b in zip(team_1_features, team_2_features)]

	prepared_data = prepare_data(matchup_features)

	return model.predict(prepared_data)

def get_teams(team_list, year):
	for i in range(len(team_list)):
			for j in range(i + 1, len(team_list)):
				if team_list[i] < team_list[j]:
					prediction = predict_winner(team_list[i], team_list[j], model, year, stat_fields)
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