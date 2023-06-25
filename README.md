# NCAA-March-Madness-ML-2023

This repository contains a Python script to predict the outcome of NCAA basketball games using machine learning algorithms from the scikit-learn package. The model is trained on historical data, capturing patterns and relationships that influence game outcomes. It uses numpy and pandas for data handling, sklearn for creating and training the model, and pickle for storing the trained model.

# Main Features
- Uses historical NCAA basketball game data to train predictive models.
- Leverages popular Python libraries such as pandas, numpy, scikit-learn, and pickle.
- Uses scikit-learn, a powerful library for machine learning in Python.
- Can be applied to new data to predict game outcomes.
- Model can be saved and loaded using pickle for future use without needing retraining.

# Features Used for Training
The machine learning model in this script is trained on the following features:

- Win Percentage: The proportion of games won to total games played.
- Strength of Schedule: A measure of the difficulty of the games played by a team.
- Glicko Rating: A method for assessing a team's strength in games of skill.
- Opponent Glicko Rating: The Glicko rating of the opposing team.
- Margin of Victory: The difference in score between the two teams at the end of a game.
- Effective Field Goal Percentage: A version of field goal percentage that adjusts for the fact that three-point field goals are worth more than two-point field goals.
- Turnover Percentage: An estimate of turnovers per 100 plays.
- Offensive Rebound Percentage: An estimate of the percentage of available offensive rebounds a player or team grabbed.
Free Throw Rate: A measure of both how often a team gets to the line (free throw attempts) and how often they make their shots (free throws made).
