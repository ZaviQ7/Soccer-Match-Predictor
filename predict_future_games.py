# predict_future_games.py

import pandas as pd
import pickle
import json
import os
from src.dixon_coles import DixonColesModel # We need this for pickle to work
from src.market_calculator import calculate_1x2, calculate_over_under, calculate_asian_handicap

def predict_upcoming_matches(model_path, fixtures_path):
    """
    Loads a trained model and predicts outcomes for a list of future matches.

    Args:
        model_path (str): The path to the saved .pkl model file.
        fixtures_path (str): The path to the CSV file with upcoming games.
    """
    # --- Load the Trained Model ---
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    # --- Load and Prepare Fixtures Data ---
    try:
        # The fixtures file only needs 'HomeTeam' and 'AwayTeam' columns
        fixtures_df = pd.read_csv(fixtures_path)
        if not all(col in fixtures_df.columns for col in ['HomeTeam', 'AwayTeam']):
            raise ValueError("Fixtures CSV must contain 'HomeTeam' and 'AwayTeam' columns.")
    except Exception as e:
        print(f"Error reading fixtures file: {e}")
        return

    # --- Generate Predictions ---
    print(f"\nGenerating predictions for {len(fixtures_df)} upcoming matches...")
    predictions = []
    for index, row in fixtures_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        try:
            goal_matrix = model.predict_goal_matrix(home_team, away_team)
            
            markets = {
                '1X2': calculate_1x2(goal_matrix),
                'OU_2.5': calculate_over_under(goal_matrix, 2.5),
                'OU_3.5': calculate_over_under(goal_matrix, 3.5),
                'AH_-0.5': calculate_asian_handicap(goal_matrix, -0.5),
                'AH_+0.5': calculate_asian_handicap(goal_matrix, 0.5),
                'AH_-1.5': calculate_asian_handicap(goal_matrix, -1.5),
                'AH_+1.5': calculate_asian_handicap(goal_matrix, 1.5),
            }
            
            predictions.append({
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'FairOdds': markets
            })
        except ValueError as e:
            print(f"Skipping prediction for {home_team} vs {away_team}: {e}")

    # --- Save Predictions ---
    output_csv_path = 'output/future_predictions.csv'
    output_json_path = 'output/future_predictions.json'

    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    csv_output = []
    for p in predictions:
        flat = {
            'HomeTeam': p['HomeTeam'], 'AwayTeam': p['AwayTeam'],
            'Prob_1': p['FairOdds']['1X2']['1']['prob'],
            'Prob_X': p['FairOdds']['1X2']['X']['prob'],
            'Prob_2': p['FairOdds']['1X2']['2']['prob'],
            'Odds_1': p['FairOdds']['1X2']['1']['odds'],
            'Odds_X': p['FairOdds']['1X2']['X']['odds'],
            'Odds_2': p['FairOdds']['1X2']['2']['odds'],
            'Prob_O2.5': p['FairOdds']['OU_2.5']['Over']['prob'],
            'Prob_U2.5': p['FairOdds']['OU_2.5']['Under']['prob'],
        }
        csv_output.append(flat)
    pd.DataFrame(csv_output).to_csv(output_csv_path, index=False)
    
    print(f"\nPredictions saved to {output_json_path} and {output_csv_path}")

if __name__ == '__main__':
    # Define the paths for the model and the new games
    trained_model_file = 'output/trained_model.pkl'
    upcoming_games_file = 'data/games_to_predict.csv'
    
    # Create a dummy fixtures file for demonstration if it doesn't exist
    if not os.path.exists(upcoming_games_file):
        print(f"Creating a dummy fixtures file at '{upcoming_games_file}'...")
        dummy_data = {'HomeTeam': ['Arsenal', 'Man City'], 'AwayTeam': ['Chelsea', 'Liverpool']}
        pd.DataFrame(dummy_data).to_csv(upcoming_games_file, index=False)

    predict_upcoming_matches(model_path=trained_model_file, fixtures_path=upcoming_games_file)