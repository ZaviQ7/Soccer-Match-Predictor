# src/utils.py

import pandas as pd

def load_data(filepath):
    """
    Loads and preprocesses football data from the detailed football-data.co.uk format.
    It selects only the necessary columns and handles potential missing data.
    """
    # Define the columns we absolutely need for the model and validation
    required_columns = [
        'Date',
        'HomeTeam',
        'AwayTeam',
        'FTHG',  # Full Time Home Goals
        'FTAG',  # Full Time Away Goals
        'PSH',   # Pinnacle Closing Home Odds
        'PSD',   # Pinnacle Closing Draw Odds
        'PSA'    # Pinnacle Closing Away Odds
    ]

    try:
        # Use 'latin1' encoding which is common for these files
        df = pd.read_csv(filepath, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Please ensure the file is a valid CSV.")
        return pd.DataFrame() # Return empty dataframe on error

    # Check if all required columns exist in the CSV
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"CSV file is missing required columns: {missing}")

    # Select only the columns we need to work with
    df = df[required_columns]

    # Drop rows where essential data (like scores or odds) is missing
    df.dropna(subset=['FTHG', 'FTAG', 'PSH', 'PSD', 'PSA'], inplace=True)

    # Convert score columns to integers
    df['FTHG'] = df['FTHG'].astype(int)
    df['FTAG'] = df['FTAG'].astype(int)

    # Convert Date column to datetime objects, using the correct format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Sort by date to ensure correct time-based train/test split
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Successfully loaded and processed {len(df)} matches.")
    return df

def get_pinnacle_implied_probs(df):
    """Calculates no-vig implied probabilities from Pinnacle odds."""
    if not all(c in df.columns for c in ['PSH', 'PSD', 'PSA']):
        print("Warning: Pinnacle odds columns (PSH, PSD, PSA) not found. Skipping benchmark.")
        return None, None, None
        
    # The rest of this function remains the same
    margin = 1/df['PSH'] + 1/df['PSD'] + 1/df['PSA']
    home_prob = (1/df['PSH']) / margin
    draw_prob = (1/df['PSD']) / margin
    away_prob = (1/df['PSA']) / margin
    return home_prob, draw_prob, away_prob