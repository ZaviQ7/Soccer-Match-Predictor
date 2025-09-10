# aggregate_data.py

import pandas as pd
import glob
import os

def aggregate_season_data():
    """
    Finds all CSV files in the 'data/historical/' directory,
    combines them into a single DataFrame, and saves it as
    'data/sample_data.csv'.
    """
    # Path to the folder containing your individual season CSVs
    historical_data_path = 'data/historical/'
    
    # The final output file required by the main model script
    output_file = 'data/sample_data.csv'

    # --- Safety Checks ---
    if not os.path.isdir(historical_data_path):
        print(f"Error: The directory '{historical_data_path}' does not exist.")
        print("Please create it and place your downloaded CSV files inside.")
        return

    # Find all files ending with .csv in the historical data folder
    all_files = glob.glob(os.path.join(historical_data_path, "*.csv"))

    if not all_files:
        print(f"Error: No CSV files were found in '{historical_data_path}'.")
        print("Please ensure your downloaded season data is in that folder.")
        return

    print(f"Found {len(all_files)} season files to aggregate.")

    # --- Aggregation Process ---
    df_list = []
    for file in all_files:
        try:
            # Read each CSV file. Using 'latin1' encoding is robust for these files.
            df = pd.read_csv(file, encoding='latin1')
            df_list.append(df)
            print(f"  - Reading {os.path.basename(file)}")
        except Exception as e:
            print(f"  - Warning: Could not read or process file {os.path.basename(file)}. Error: {e}")

    if not df_list:
        print("\nNo data could be read. The output file was not created.")
        return

    # Combine all the individual dataframes into one large dataframe
    # ignore_index=True creates a new clean index for the combined file
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the final, aggregated dataframe to the target file
    # index=False prevents pandas from adding an extra index column to the CSV
    combined_df.to_csv(output_file, index=False)

    print(f"\n✅ Success! All files have been combined into '{output_file}'.")
    print(f"Total matches aggregated: {len(combined_df)}")

if __name__ == '__main__':
    aggregate_season_data()