import pandas as pd
import numpy as np
from numpy import newaxis
import pickle
import random
import math
import csv
from dataclasses import dataclass


@dataclass
class DataParams:
	home_win_weight: float = 0.6
	away_win_weight: float = 1.4
	neutral_weight: float = 1.0
	glicko_tau: float = 0.5
	glicko_initial_rating: float = 1500
	glicko_carryover_denom: float = 200
	mov_cap: int = 10
	lookback_n: int = 5

DEFAULT_PARAMS = DataParams()


team_csv_path = '../data/2023/MTeams.csv'
season_csv_path = '../data/2023/MRegularSeasonDetailedResults.csv'
tournament_csv_path = '../data/2023/MNCAATourneyDetailedResults.csv'

start_year = 2003
prediction_year = 2023

training_data = []
train_X = []
train_y = []

class Team:
	def __init__(self, team_name, team_id, season, params=None, team_stats_ref=None):
		self.team_name = team_name
		self.team_id = team_id
		self.season = season
		self.params = params or DEFAULT_PARAMS
		self._team_stats_ref = team_stats_ref
		self.wins = 0
		self.losses = 0
		self.games_played = 0
		self.points_scored = []
		self.points_allowed = []
		self.fg_made = []
		self.fg_attempts = []
		self.fg3_made = []
		self.fg3_attempts = []
		self.ft_made = []
		self.ft_attempts = []
		self.off_rebounds = []
		self.def_rebounds = []
		self.assists = []
		self.turnovers = []
		self.steals = []
		self.blocks = []
		self.fouls = []
		self.glicko_rating = [self.calculate_new_season_glicko_rating()]
		self.glicko_deviation = [400]
		self.opponents = []
		self.offensive_efficiency = []
		self.defensive_efficiency = []
		self.opponents_def_rebounds = []
		self.opponents_off_rebounds = []
		
	def add_game(self, opponent, score, opponent_score, fgm, fga, fgm3, fga3, ftm, fta, off_rebounds, def_rebounds, assists, turnovers, steals, blocks, fouls, loc, opp_def_rebounds, opp_off_rebounds):
		self.games_played += 1
		self.points_scored.append(score)
		self.points_allowed.append(opponent_score)
		self.opponents.append(opponent)
		self.fg_made.append(fgm)
		self.fg_attempts.append(fga)
		self.fg3_made.append(fgm3)
		self.fg3_attempts.append(fga3)
		self.ft_made.append(ftm)
		self.ft_attempts.append(fta) 
		self.off_rebounds.append(off_rebounds)
		self.def_rebounds.append(def_rebounds)
		self.assists.append(assists)
		self.turnovers.append(turnovers)
		self.steals.append(steals)
		self.blocks.append(blocks)
		self.fouls.append(fouls)
		self.offensive_efficiency.append(self.get_offensive_efficiency())
		self.defensive_efficiency.append(self.get_defensive_efficiency())
		self.opponents_def_rebounds.append(opp_def_rebounds)
		self.opponents_off_rebounds.append(opp_off_rebounds)
		if score > opponent_score:
			self.calculate_glicko_rating(opponent, 1)
			if loc == 'H':
				self.wins += self.params.home_win_weight
			elif loc == 'A':
				self.wins += self.params.away_win_weight
			else:
				self.wins += self.params.neutral_weight
		else:
			self.calculate_glicko_rating(opponent, 0)
			if loc == 'H':
				self.losses += self.params.home_win_weight
			elif loc == 'A':
				self.losses += self.params.away_win_weight
			else:
				self.losses += self.params.neutral_weight

	def get_win_percentage(self):
		if self.games_played == 0:
			return 0 
		return self.wins / (self.wins + self.losses)
	
	def get_points_per_game(self):
		if self.games_played == 0:
			return 0
		return sum(self.points_scored) / self.games_played
	
	def get_points_allowed_per_game(self):
		if self.games_played == 0:
			return 0
		return sum(self.points_allowed) / self.games_played
	
	def get_opponent_win_percentage(self):
		if len(self.opponents) == 0:
			return 0
		return sum([opponent.get_win_percentage() for opponent in self.opponents]) / len(self.opponents)
	
	def get_opponent_points_allowed_per_game(self):
		if len(self.opponents) == 0:
			return 0
		return sum([opponent.get_points_allowed_per_game() for opponent in self.opponents]) / len(self.opponents)

	def get_turnover_percentage(self):
		return sum(self.turnovers) / (sum(self.fg_attempts) + (0.44 * sum(self.ft_attempts)) + sum(self.turnovers)) * 100

	def get_free_throw_rate(self):
		if sum(self.ft_made) == 0:
			return 0
		else:
			return sum(self.ft_made) / sum(self.ft_attempts)

	def get_offensive_efficiency(self):
		if self.games_played == 0:
			return 0

		num_of_possessions = .96 * self.fg_attempts[-1] - (self.off_rebounds[-1] + self.turnovers[-1] + (.475 * self.ft_attempts[-1]))
		try:
			offensive_efficiency = (self.points_scored[-1]*100)/num_of_possessions
		except:
			offensive_efficiency = 0
		
		return offensive_efficiency

	def get_defensive_efficiency(self):
		if self.games_played == 0:
			return 0

		try:
			opponent = self.opponents[-1]
			opp_fg_attempts = opponent.fg_attempts[-1]
			opp_off_rebounds = opponent.off_rebounds[-1]
			opp_turnovers = opponent.turnovers[-1]
			opp_ft_attempts = opponent.ft_attempts[-1]
			opp_score = opponent.points_scored[-1]

			num_of_possessions = .96*(opp_fg_attempts - opp_off_rebounds + opp_turnovers + (.475 * opp_ft_attempts))
			try:
				defensive_efficiency = (opp_score*100)/num_of_possessions
			except:
				pass
		except:
			defensive_efficiency = 0
		
		return defensive_efficiency

	def calculate_strength_of_schedule(self):
		if self.games_played == 0:
			return 0
		
		opponent_win_percentage = sum([opponent.get_win_percentage() for opponent in self.opponents]) / len(self.opponents)

		total_opponent_opponent_win_percentage = []
		for opponent in self.opponents: 
			for opponent_opponent in opponent.opponents:
				total_opponent_opponent_win_percentage.append(opponent_opponent.get_win_percentage())

		opponent_opponent_win_percentage = sum(total_opponent_opponent_win_percentage) / len(total_opponent_opponent_win_percentage)

		return ((2 * opponent_win_percentage) + opponent_opponent_win_percentage) / 3

	def calculate_new_season_glicko_rating(self):
		if self.season == start_year:
			return self.params.glicko_initial_rating

		stats_ref = self._team_stats_ref if self._team_stats_ref is not None else team_stats
		try:
			for team in stats_ref[self.season - 1]:
				if team.team_id == self.team_id:
					last_season_glicko = team.glicko_rating[-1]
					last_season_deviation = team.glicko_deviation[-1]
					carry_over_percentage = 1 - (last_season_deviation / (last_season_deviation + self.params.glicko_carryover_denom))
					return last_season_glicko * carry_over_percentage
		except:
			return self.params.glicko_initial_rating


	def calculate_glicko_rating(self, opponent, outcome):
		team_rating = self.glicko_rating[-1]
		team_deviation = self.glicko_deviation[-1]
		opponent_rating = opponent.glicko_rating[-1]
		opponent_deviation = opponent.glicko_deviation[-1]

		if opponent_rating == None:
			opponent_rating = self.params.glicko_initial_rating
			opponent_deviation = 400

		if team_rating == None:
			team_rating = self.params.glicko_initial_rating
			team_deviation = 400

		tau = self.params.glicko_tau

		rating_difference = team_rating - opponent_rating

		g = 1 / math.sqrt(1 + ((3 * math.pow(opponent_deviation, 2))/math.pow(math.pi, 2)))
		e = 1 / (1 + math.exp(-g * rating_difference))

		v = math.pow((math.pow(g, 2) * e) * (1 - e), -1)

		delta = (v * g) * (outcome - e)

		a = math.log(math.pow(team_deviation, 2))

		x_0 = a
		x_1 = 0

		delta_squared = math.pow(delta, 2)
		tau_squared = math.pow(tau, 2)
		v_squared = math.pow(v, 2)

		while math.fabs(x_0 - x_1) > 0.000001:
			d = delta_squared + v + math.exp(x_0)
			h_0 = -x_0 + a + (tau_squared * v_squared * d) / (1 + v_squared * d)
			h_1 = -1 + (tau_squared * v_squared) / (1 + v_squared * d)
			x_1 = x_0
			x_0 = x_0 - h_0 / h_1

		volatility = math.pow(x_0, 2)

		new_deviation = 1 / math.sqrt((1 / (math.pow(opponent_deviation, 2) + math.pow(volatility, 2)))+(1 / v))
		new_rating = team_rating + ((math.pow(new_deviation, 2) * g) * (outcome - e))
		self.glicko_rating.append(new_rating)
		self.glicko_deviation.append(new_deviation)

	def get_opponent_glicko_last_n(self, n):
		if self.games_played == 0:
			return 0

		try:
			last_n_opponents = self.opponents[-n:]
			return sum([opponent.glicko_rating[-1] for opponent in last_n_opponents])/n
		except:
			return sum([opponent.glicko_rating[-1] for opponent in self.opponents]) / len(self.opponents)

	def get_margin_of_victory_last_n(self, n):
		if self.games_played == 0:
			return 0

		try:
			last_n_points = self.points_scored[-n:]
			last_n_points_allowed = self.points_allowed[-n:]

			mov = []

			for x in range(n):
				i = last_n_points[x] - last_n_points_allowed[x]

				if i > self.params.mov_cap:
					i = self.params.mov_cap

				mov.append(i)

			return sum(mov) / len(mov)

		except:

			mov = []

			for x in range(self.games_played):
				n = self.points_scored[x] - self.points_allowed[x]

				if n > self.params.mov_cap:
					n = self.params.mov_cap

				mov.append(n)

			return sum(mov) / len(mov)

	def get_efficiency_last_n(self, n):
		if self.games_played == 0:
			return 0

		try:
			last_5_offensive_efficiency = self.offensive_efficiency[-n:]
			last_5_defensive_efficiency = self.defensive_efficiency[-n:]

			efficiency = []

			for x in range(n):
				i = last_5_offensive_efficiency[x] - last_5_defensive_efficiency[x]
				efficiency.append(i)

			return sum(efficiency) / len(efficiency)

		except:
			efficiency = []

			for x in range(self.games_played):
				# print(self.offensive_efficiency)
				# print(self.defensive_efficiency)
				i = self.offensive_efficiency[x] - self.defensive_efficiency[x]
				efficiency.append(i)

			return sum(efficiency) / len(efficiency)

	def get_effective_field_goal_percentage_last_n(self, n):
		if self.games_played == 0:
			return 0

		try:
			last_n_fgm = self.fg_made[-n:]
			last_n_fgm3 = self.fg3_made[-n:]
			last_n_fga = self.fg_attempts[-n:]

			efgp = []

			for x in range(n):
				i = (last_n_fgm[x] + (0.5 * last_n_fgm3[x])) / last_n_fga[x]
				efgp.append(i)

			return sum(efgp) / len(efgp)

		except:
			efgp = []

			for x in range(self.games_played):
				i = (self.fg_made[x] + (0.5 * self.fg3_made[x])) / self.fg_attempts[x]
				efgp.append(i)

			return sum(efgp) / len(efgp)

	def get_turnover_percentage_last_n(self, n):
		if self.games_played == 0:
			return 0

		try:
			last_n_fga = self.fg_attempts[-n:]
			last_n_off_rebounds = self.off_rebounds[-n:]
			last_n_turnovers = self.turnovers[-n:]
			last_n_ft_attempts = self.ft_attempts[-n:]

			turnover_percentages = []

			for x in range(n):
				i = last_n_turnovers[x] / (.96 * (last_n_fga[x] - last_n_off_rebounds[x] + last_n_turnovers[x] + (.475 * last_n_ft_attempts[x])))
				turnover_percentages.append(i)

			return sum(turnover_percentages) / len(turnover_percentages)

		except:
			turnover_percentages = []

			for x in range(self.games_played):
				i = self.turnovers[x] / (.96 * (self.fg_attempts[x] - self.off_rebounds[x] + self.turnovers[x] + (.475 * self.ft_attempts[x])))
				turnover_percentages.append(i)

			return sum(turnover_percentages) / len(turnover_percentages)

	def get_offensive_rebound_percentage_last_n(self, n):
		if self.games_played == 0:
			return 0

		try:
			last_n_off_rebounds = self.off_rebounds[-n:]
			last_n_opponents_def_rebounds = self.opponents_def_rebounds[-n:]

			offensive_rebound_percentages = []

			for x in range(n):
				i = last_n_off_rebounds[x] / (last_n_off_rebounds[x] + last_n_opponents_def_rebounds[x])
				offensive_rebound_percentages.append(i)

			return sum(offensive_rebound_percentages) / len(offensive_rebound_percentages)

		except:
			offensive_rebound_percentages = []

			for x in range(self.games_played):
				i = self.off_rebounds[x] / (self.off_rebounds[x] + self.opponents_def_rebounds[x])
				offensive_rebound_percentages.append(i)

			return sum(offensive_rebound_percentages) / len(offensive_rebound_percentages)

	def get_freethrow_rate_last_n(self, n):
		if self.games_played == 0:
			return 0

		try:
			last_n_fta = self.ft_attempts[-n:]
			last_n_fga = self.fg_attempts[-n:]

			freethrow_rates = []

			for x in range(n):
				i = last_n_fta[x] / last_n_fga[x]
				freethrow_rates.append(i)

			return sum(freethrow_rates) / len(freethrow_rates)

		except:
			freethrow_rates = []

			for x in range(self.games_played):
				i = self.ft_attempts[x] / self.fg_attempts[x]
				freethrow_rates.append(i)

			return sum(freethrow_rates) / len(freethrow_rates)

	def get_offensive_efficiency_last_n(self, n):
		if self.games_played == 0:
			return 0
		n = min(n, self.games_played)
		return sum(self.offensive_efficiency[-n:]) / n

	def get_defensive_efficiency_last_n(self, n):
		if self.games_played == 0:
			return 0
		n = min(n, self.games_played)
		return sum(self.defensive_efficiency[-n:]) / n

	def get_three_point_rate_last_n(self, n):
		if self.games_played == 0:
			return 0
		n = min(n, self.games_played)
		total_fga = sum(self.fg_attempts[-n:])
		if total_fga == 0:
			return 0
		return sum(self.fg3_attempts[-n:]) / total_fga

	def get_assist_to_turnover_ratio_last_n(self, n):
		if self.games_played == 0:
			return 0
		n = min(n, self.games_played)
		total_to = sum(self.turnovers[-n:])
		if total_to == 0:
			return 0
		return sum(self.assists[-n:]) / total_to

	def get_defensive_rebound_percentage_last_n(self, n):
		if self.games_played == 0:
			return 0
		n = min(n, self.games_played)
		last_n_dr = self.def_rebounds[-n:]
		last_n_opp_or = self.opponents_off_rebounds[-n:]
		rates = []
		for dr, opp_or in zip(last_n_dr, last_n_opp_or):
			denom = dr + opp_or
			if denom > 0:
				rates.append(dr / denom)
		return sum(rates) / len(rates) if rates else 0

	def get_steal_rate_last_n(self, n):
		if self.games_played == 0:
			return 0
		n = min(n, self.games_played)
		return sum(self.steals[-n:]) / n

	def get_block_rate_last_n(self, n):
		if self.games_played == 0:
			return 0
		n = min(n, self.games_played)
		return sum(self.blocks[-n:]) / n

	def get_features(self, n):
		team_glicko_rating = self.glicko_rating[-1]

		if team_glicko_rating == None:
			team_glicko_rating = self.params.glicko_initial_rating

		features = [
			self.get_win_percentage(),
			self.calculate_strength_of_schedule(),
			team_glicko_rating,
			self.get_opponent_glicko_last_n(n),
			self.get_margin_of_victory_last_n(n),
			self.get_offensive_efficiency_last_n(n),
			self.get_defensive_efficiency_last_n(n),
			self.get_effective_field_goal_percentage_last_n(n),
			self.get_turnover_percentage_last_n(n),
			self.get_offensive_rebound_percentage_last_n(n),
			self.get_defensive_rebound_percentage_last_n(n),
			self.get_freethrow_rate_last_n(n),
			self.get_three_point_rate_last_n(n),
			self.get_assist_to_turnover_ratio_last_n(n),
			self.get_steal_rate_last_n(n),
			self.get_block_rate_last_n(n),
		]

		return features


def initiate_data(start_year, prediction_year):
	team_stats = {}

	for season in range (start_year, prediction_year+1):
		team_stats[season] = []

	return team_stats

def get_team(season, team_name, team_id):
	for team in team_stats[season]:
		if team.team_id == team_id:
			return team
	team = Team(team_name, team_id, season, team_stats_ref=team_stats)
	team_stats[season].append(team)
	return team

def create_team_array(csv_filename):
	team_array = []
	
	df = pd.read_csv(csv_filename)
	
	for index, row in df.iterrows():
		team_name = row['TeamName']
		team_id = row['TeamID']
		
		team = Team(team_name, team_id)
		team_array.append(team)
	
	return team_array

def glicko_expected_outcome(team_1, team_2):
	team_1_glicko_rating = team_1.glicko_rating[-1]
	team_2_glicko_rating = team_2.glicko_rating[-1]
	team_2_glicko_deviation = team_2.glicko_deviation[-1]

	rating_difference = team_1_glicko_rating - team_2_glicko_rating

	g = -rating_difference/400
	e = 1 / (1 + math.pow(10, g))

	return e

def build_season_data(all_data, team_names):
	print('Simulating seasons...')

	for index, row in all_data.iterrows():
		season = row['Season']

		if season < start_year:
			continue

		win_team_id = row['WTeamID']
		lose_team_id = row['LTeamID']
		loc = row['WLoc']
		daynum = row['DayNum']

		win_team_name = team_names.loc[team_names['TeamID'] == win_team_id].values[0][1]
		lose_team_name = team_names.loc[team_names['TeamID'] == lose_team_id].values[0][1]

		win_team = get_team(season, win_team_name, win_team_id)
		lose_team = get_team(season, lose_team_name, lose_team_id)

		winner_features = win_team.get_features(5)

		loser_features = lose_team.get_features(5)

		win_team.add_game(
			lose_team,
			row['WScore'],
			row['LScore'],
			row['WFGM'],
			row['WFGA'],
			row['WFGM3'],
			row['WFGA3'],
			row['WFTM'],
			row['WFTA'],
			row['WOR'],
			row['WDR'],
			row['WAst'],
			row['WTO'],
			row['WStl'],
			row['WBlk'],
			row['WPF'],
			loc,
			row['LDR'],
			row['LOR'])
		lose_team.add_game(
			win_team,
			row['LScore'],
			row['WScore'],
			row['LFGM'],
			row['LFGA'],
			row['LFGM3'],
			row['LFGA3'],
			row['LFTM'],
			row['LFTA'],
			row['LOR'],
			row['LDR'],
			row['LAst'],
			row['LTO'],
			row['LStl'],
			row['LBlk'],
			row['LPF'],
			loc,
			row['WDR'],
			row['WOR'])

		matchup_features = []

		if index % 2 == 0:
			for i in range(len(winner_features)):
				matchup_features.append(winner_features[i] - loser_features[i])
			training_data.append([matchup_features, 1])
		else:
			for i in range(len(winner_features)):
				matchup_features.append(loser_features[i] - winner_features[i])
			training_data.append([matchup_features, 0])


def build_season_data_parameterized(all_data, team_names, sy=2003, py=2023, params=None):
	"""Build features with given params. Returns (X, y, team_stats_local)."""
	if params is None:
		params = DEFAULT_PARAMS

	team_stats_local = {}
	for season in range(sy, py + 1):
		team_stats_local[season] = []

	def get_team_local(season, team_name, team_id):
		for team in team_stats_local[season]:
			if team.team_id == team_id:
				return team
		team = Team(team_name, team_id, season, params=params, team_stats_ref=team_stats_local)
		team_stats_local[season].append(team)
		return team

	n = params.lookback_n
	training_data_local = []

	for index, row in all_data.iterrows():
		season = row['Season']
		if season < sy:
			continue

		win_team_id = row['WTeamID']
		lose_team_id = row['LTeamID']
		loc = row['WLoc']

		win_team_name = team_names.loc[team_names['TeamID'] == win_team_id].values[0][1]
		lose_team_name = team_names.loc[team_names['TeamID'] == lose_team_id].values[0][1]

		win_team = get_team_local(season, win_team_name, win_team_id)
		lose_team = get_team_local(season, lose_team_name, lose_team_id)

		winner_features = win_team.get_features(n)
		loser_features = lose_team.get_features(n)

		win_team.add_game(
			lose_team, row['WScore'], row['LScore'],
			row['WFGM'], row['WFGA'], row['WFGM3'], row['WFGA3'],
			row['WFTM'], row['WFTA'], row['WOR'], row['WDR'],
			row['WAst'], row['WTO'], row['WStl'], row['WBlk'], row['WPF'],
			loc, row['LDR'], row['LOR'])
		lose_team.add_game(
			win_team, row['LScore'], row['WScore'],
			row['LFGM'], row['LFGA'], row['LFGM3'], row['LFGA3'],
			row['LFTM'], row['LFTA'], row['LOR'], row['LDR'],
			row['LAst'], row['LTO'], row['LStl'], row['LBlk'], row['LPF'],
			loc, row['WDR'], row['WOR'])

		matchup_features = []
		if index % 2 == 0:
			for i in range(len(winner_features)):
				matchup_features.append(winner_features[i] - loser_features[i])
			training_data_local.append([matchup_features, 1])
		else:
			for i in range(len(winner_features)):
				matchup_features.append(loser_features[i] - winner_features[i])
			training_data_local.append([matchup_features, 0])

	random.seed(42)
	random.shuffle(training_data_local)

	X = [f for f, _ in training_data_local]
	y = [l for _, l in training_data_local]
	return np.array(X), np.array(y), team_stats_local


team_stats = initiate_data(start_year, prediction_year)

season_data = pd.read_csv(season_csv_path)
tournament_data = pd.read_csv(tournament_csv_path)
team_names = pd.read_csv(team_csv_path)

all_data = pd.concat([season_data, tournament_data], ignore_index=True)
all_data = all_data.sort_values(by=['Season', 'DayNum'])

build_season_data(all_data, team_names)
random.shuffle(training_data)

for features, label in training_data:
	train_X.append(features)
	train_y.append(label)

pickle_out = open("../Pickles/X.pickle", "wb")
pickle.dump(train_X, pickle_out)
pickle_out.close()

pickle_out = open("../Pickles/y.pickle", "wb")
pickle.dump(train_y, pickle_out)
pickle_out.close()

pickle_out = open("../Pickles/team_stats.pickle", "wb")
pickle.dump(team_stats, pickle_out)
pickle_out.close()

team_seeds = {}
team_list = []
final_data = []
prediction_range = [2023]