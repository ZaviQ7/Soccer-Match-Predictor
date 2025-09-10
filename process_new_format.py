# process_new_format.py

import pandas as pd
import numpy as np
import os

def process_fixture_list(input_file_path, historical_output_path, future_output_path):
    """
    Reads a CSV with a specific fixture format, cleans it, and splits it into
    historical results and future games to predict.
    """
    print(f"Reading new fixture file: {input_file_path}")
    
    try:
        # Read the source file
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
        return

    # --- Data Cleaning and Formatting ---

    # 1. Rename columns to match our model's expected format
    df.rename(columns={
        'Home Team': 'HomeTeam',
        'Away Team': 'AwayTeam'
    }, inplace=True)

    # 2. Convert the 'Date' column to datetime objects
    # The errors='coerce' will turn any unparseable dates into NaT (Not a Time)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', errors='coerce')

    # 3. Split the 'Result' column into Home and Away Goals
    # .str.split() returns a Series of lists, expand=True puts them in separate columns
    # We use .get(0) and .get(1) to safely access the elements, avoiding errors
    score_split = df['Result'].str.split(' - ', expand=True)
    df['FTHG'] = pd.to_numeric(score_split.get(0), errors='coerce')
    df['FTAG'] = pd.to_numeric(score_split.get(1), errors='coerce')

    # --- Splitting Data ---

    # 1. Create a dataframe for games that have already been played (where FTHG is not null)
    historical_df = df[df['FTHG'].notna()].copy()
    
    # 2. Create a dataframe for future games to predict (where FTHG is null)
    future_df = df[df['FTHG'].isna()].copy()

    # --- Preparing and Saving Files ---

    # 1. Save the future games file (only needs HomeTeam and AwayTeam)
    if not future_df.empty:
        future_to_save = future_df[['HomeTeam', 'AwayTeam']]
        future_to_save.to_csv(future_output_path, index=False)
        print(f"✅ Successfully created fixtures file with {len(future_to_save)} games at: {future_output_path}")
    else:
        print("No future games found in the input file.")

    # 2. Save the historical games file
    # Note: This file will NOT have odds columns (PSH, PSD, PSA)
    if not historical_df.empty:
        # Select and reorder columns to match the training data format as closely as possible
        historical_to_save = historical_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
        
        # Add placeholder columns for Pinnacle odds since they don't exist in the source
        for col in ['PSH', 'PSD', 'PSA']:
            historical_to_save[col] = np.nan

        historical_to_save.to_csv(historical_output_path, index=False)
        print(f"✅ Successfully created a processed historical file with {len(historical_to_save)} results at: {historical_output_path}")
    else:
        print("No historical results found in the input file.")


if __name__ == '__main__':
    # Define the input file from the user
    new_file = 'data/historical/epl-2025-GMTStandardTime.csv'
    
    # Define the output files
    processed_historical_file = 'data/historical/processed_25-26_season.csv'
    games_to_predict_file = 'data/games_to_predict.csv'
    
    process_fixture_list(
        input_file_path=new_file,
        historical_output_path=processed_historical_file,
        future_output_path=games_to_predict_file
    )
    print("\nNext steps:")
    print("1. Run 'python aggregate_data.py' to combine the new historical data.")
    print("2. Run 'python main.py' to retrain the model on all data.")
    print("3. Run 'python predict_future_games.py' to get odds for the upcoming matches.")