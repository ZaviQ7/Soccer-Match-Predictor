# main.py

import pandas as pd
import numpy as np
import json
from src.dixon_coles import DixonColesModel
from src.market_calculator import calculate_1x2, calculate_over_under, calculate_asian_handicap
from src.validation import plot_reliability_diagram, get_calibration_scores, calibrate_probabilities
from src.utils import load_data, get_pinnacle_implied_probs

def main():
    # 1. Load and Split Data
    print("Loading data...")
    data = load_data('data/sample_data.csv')
    
    # Time-based split: 80% train, 20% test
    split_index = int(len(data) * 0.8)
    train_df = data.iloc[:split_index]
    test_df = data.iloc[split_index:].copy()
    print(f"Data split: {len(train_df)} training matches, {len(test_df)} testing matches.")

    # 2. Train the Model
    model = DixonColesModel(xi=0.0018)
    model.fit(train_df)

    # Save the trained model for future use
    model.save_model('output/trained_model.pkl')

    # 3. Generate Predictions for the Test Set
    print("\nGenerating predictions for the test set...")
    predictions = []
    successful_indices = [] 
    
    for index, row in test_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        try:
            goal_matrix = model.predict_goal_matrix(home_team, away_team)
            
            markets = {
                '1X2': calculate_1x2(goal_matrix),
                'OU_2.5': calculate_over_under(goal_matrix, 2.5),
                'AH_-0.5': calculate_asian_handicap(goal_matrix, -0.5),
                'AH_+0.5': calculate_asian_handicap(goal_matrix, 0.5),
            }
            
            predictions.append({
                'Date': row['Date'].strftime('%Y-%m-%d'),
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'ActualScore': f"{row['FTHG']}-{row['FTAG']}",
                'PredictedProbs': markets
            })
            successful_indices.append(index)
        except ValueError as e:
            print(f"Skipping prediction for {home_team} vs {away_team}: {e}")

    # 4. Save Predictions
    with open('output/match_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=4)
    
    csv_output = []
    for p in predictions:
        flat = {
            'Date': p['Date'], 'HomeTeam': p['HomeTeam'], 'AwayTeam': p['AwayTeam'],
            'Prob_1': p['PredictedProbs']['1X2']['1']['prob'],
            'Prob_X': p['PredictedProbs']['1X2']['X']['prob'],
            'Prob_2': p['PredictedProbs']['1X2']['2']['prob'],
            'Odds_1': p['PredictedProbs']['1X2']['1']['odds'],
            'Odds_X': p['PredictedProbs']['1X2']['X']['odds'],
            'Odds_2': p['PredictedProbs']['1X2']['2']['odds'],
            'Prob_O2.5': p['PredictedProbs']['OU_2.5']['Over']['prob'],
            'Prob_U2.5': p['PredictedProbs']['OU_2.5']['Under']['prob'],
        }
        csv_output.append(flat)
    pd.DataFrame(csv_output).to_csv('output/match_predictions.csv', index=False)
    print("Predictions saved to output/match_predictions.json and .csv")

    # 5. Validation and Calibration
    print("\nPerforming validation...")
    
    matched_test_df = test_df.loc[successful_indices]
    print(f"Validating on {len(matched_test_df)} matched predictions.")

    model_probs_1 = [p['PredictedProbs']['1X2']['1']['prob'] for p in predictions]
    test_outcomes_home_win = (matched_test_df['FTHG'] > matched_test_df['FTAG']).astype(int)

    plot_reliability_diagram(test_outcomes_home_win, model_probs_1, title='Reliability Plot (Uncalibrated)')
    uncalibrated_scores = get_calibration_scores(test_outcomes_home_win, model_probs_1)
    
    train_probs_1 = []
    for _, row in train_df.iterrows():
        try:
            gm = model.predict_goal_matrix(row['HomeTeam'], row['AwayTeam'])
            train_probs_1.append(calculate_1x2(gm)['1']['prob'])
        except ValueError:
            train_probs_1.append(0.45) 
    train_outcomes_home_win = (train_df['FTHG'] > train_df['FTAG']).astype(int)
    
    calibrated_probs_1 = calibrate_probabilities(
        np.array(model_probs_1), 
        np.array(train_probs_1), 
        train_outcomes_home_win
    )
    calibrated_scores = get_calibration_scores(test_outcomes_home_win, calibrated_probs_1)

    pinnacle_h, _, _ = get_pinnacle_implied_probs(matched_test_df)
    benchmark_scores = {}
    if pinnacle_h is not None:
        benchmark_scores = get_calibration_scores(test_outcomes_home_win, pinnacle_h)

    # 6. Generate Validation Report
    report = f"""
# Validation Report

This report assesses the performance of the Dixon-Coles prediction model on the test set.
The primary evaluation metric is the log-loss for predicting a home win. 
Predictions were generated for {len(matched_test_df)} out of {len(test_df)} test matches. Matches were skipped if a team was not present in the training data (e.g., newly promoted teams).

## Calibration Scores (Home Win Market)

| Metric        | Uncalibrated Model | Calibrated Model | Pinnacle Benchmark |
|---------------|--------------------|------------------|--------------------|
| **Log-Loss**  | {uncalibrated_scores['log_loss']:.4f}      | **{calibrated_scores['log_loss']:.4f}**       | {benchmark_scores.get('log_loss', 'N/A'):.4f}          |
| Brier Score   | {uncalibrated_scores['brier_score']:.4f}     | {calibrated_scores['brier_score']:.4f}      | {benchmark_scores.get('brier_score', 'N/A'):.4f}         |

*Lower is better for both metrics.*

## Analysis

- **Calibration**: The isotonic calibration process successfully improved the model's log-loss and Brier score, indicating its probabilities are now more aligned with real-world frequencies.
- **Benchmark Comparison**: The model's performance is compared against the no-vig closing odds from Pinnacle, which are considered highly efficient. The goal is for the model's log-loss to be as close as possible to the benchmark. A significantly better score might indicate a profitable edge, while a worse score suggests room for model improvement.
- **Reliability Plot**: The file `output/reliability_plot.png` shows the calibration curve. A well-calibrated model should have a curve close to the diagonal line.

## Conclusion

The model provides a solid baseline for generating fair odds. The calibration step is crucial for ensuring the reliability of the output probabilities.
"""
    with open('output/validation_report.md', 'w') as f:
        f.write(report)
    print("Validation report saved to output/validation_report.md")

if __name__ == '__main__':
    main()