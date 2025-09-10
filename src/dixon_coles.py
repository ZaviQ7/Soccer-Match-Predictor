# src/dixon_coles.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import pickle

class DixonColesModel:
    """
    Implements the Dixon-Coles model for predicting football match outcomes
    with time-weighting for recency.

    Attributes:
        teams (list): List of unique team names.
        attack_params (dict): Trained attack strength for each team.
        defence_params (dict): Trained defence strength for each team.
        home_advantage (float): Trained home advantage parameter.
        rho (float): Trained goal correlation parameter.
    """
    def __init__(self, xi=0.0018):
        """
        Initializes the model.
        
        Args:
            xi (float): Time decay parameter for weighting recent matches more heavily.
                        A common value is ~0.0018, corresponding to a half-life of 1 year.
        """
        self.xi = xi
        self.teams = []
        self.attack_params = {}
        self.defence_params = {}
        self.home_advantage = 0
        self.rho = 0

    def _tau(self, i, j, lambda_val, mu_val, rho):
        """Dixon-Coles adjustment function for low-scoring results."""
        if i == 0 and j == 0:
            return 1 - lambda_val * mu_val * rho
        elif i == 0 and j == 1:
            return 1 + lambda_val * rho
        elif i == 1 and j == 0:
            return 1 + mu_val * rho
        elif i == 1 and j == 1:
            return 1 - rho
        else:
            return 1.0

    def _log_likelihood(self, params, data):
        """
        Calculates the negative log-likelihood for the given parameters and data.
        This is the function to be minimized.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        num_teams = len(self.teams)
        attack = params[0:num_teams]
        defence = params[num_teams:2*num_teams]
        home_advantage = params[2*num_teams]
        rho = params[2*num_teams + 1]

        log_likelihood_sum = 0
        
        team_map = {team: i for i, team in enumerate(self.teams)}

        for index, row in data.iterrows():
            home_team_idx = team_map[row['HomeTeam']]
            away_team_idx = team_map[row['AwayTeam']]
            
            time_weight = np.exp(-self.xi * row['TimeDiff'])

            lambda_val = np.exp(attack[home_team_idx] + defence[away_team_idx] + home_advantage)
            mu_val = np.exp(attack[away_team_idx] + defence[home_team_idx])

            tau_val = self._tau(row['FTHG'], row['FTAG'], lambda_val, mu_val, rho)
            
            log_likelihood_match = time_weight * (
                np.log(tau_val) +
                poisson.logpmf(row['FTHG'], lambda_val) +
                poisson.logpmf(row['FTAG'], mu_val)
            )
            
            log_likelihood_sum += log_likelihood_match

        return -log_likelihood_sum

    def fit(self, train_data):
        """
        Trains the model on the provided training data.
        """
        self.teams = sorted(list(pd.unique(train_data[['HomeTeam', 'AwayTeam']].values.ravel('K'))))
        num_teams = len(self.teams)

        train_data['TimeDiff'] = (train_data['Date'].max() - train_data['Date']).dt.days
        
        initial_params = np.concatenate([
            np.ones(num_teams) * 0.1,
            np.ones(num_teams) * -0.1,
            [0.2],
            [0.1]
        ])

        bounds = [(None, None)] * (2 * num_teams) + [(0, 1), (-1, 1)]

        print("Starting model optimization...")
        result = minimize(
            self._log_likelihood,
            initial_params,
            args=(train_data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True, 'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Warning: Optimization failed. Message: {result.message}")

        team_map = {team: i for i, team in enumerate(self.teams)}
        self.attack_params = {team: result.x[team_map[team]] for team in self.teams}
        self.defence_params = {team: result.x[num_teams + team_map[team]] for team in self.teams}
        self.home_advantage = result.x[2 * num_teams]
        self.rho = result.x[2 * num_teams + 1]
        print("Model training complete.")

    def predict_goal_matrix(self, home_team, away_team, max_goals=6):
        """
        Predicts the probability distribution of goals for a given match.
        """
        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError("One or both teams not found in the trained model.")

        home_attack = self.attack_params[home_team]
        home_defence = self.defence_params[home_team]
        away_attack = self.attack_params[away_team]
        away_defence = self.defence_params[away_team]

        lambda_val = np.exp(home_attack + away_defence + self.home_advantage)
        mu_val = np.exp(away_attack + home_defence)

        goal_matrix = np.zeros((max_goals + 1, max_goals + 1))

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau_val = self._tau(i, j, lambda_val, mu_val, self.rho)
                prob = tau_val * poisson.pmf(i, lambda_val) * poisson.pmf(j, mu_val)
                goal_matrix[i, j] = prob
        
        goal_matrix /= np.sum(goal_matrix)
        
        return goal_matrix

    def save_model(self, filepath):
        """Saves the trained model object to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")