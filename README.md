# Football Fair-Odds Engine

This project provides a complete, end-to-end Python engine for generating fair odds for football matches. It uses a time-weighted Dixon-Coles model to estimate goal distributions and then calculates no-vig fair odds for several popular betting markets. The model's performance is rigorously validated against Pinnacle's closing odds to ensure accuracy and reliability.

## Key Features

- **Goals-Based Modeling**: Implements the Dixon-Coles model to calculate team-specific attack and defence strengths.
- **Dynamic Parameters**: The model accounts for **Home Advantage** and **Recency/Form** through a time-decay function that weights recent matches more heavily.
- **Multiple Market Outputs**: Generates probabilities and fair odds for:
    - **1X2 (Match Winner)**
    - **Over/Under Totals** (e.g., O/U 2.5)
    - **Asian Handicap** (e.g., AH -0.5, +1.5)
- **Rigorous Validation**:
    - Employs a strict **time-based train/test split** to prevent data leakage.
    - Benchmarks performance against **Pinnacle's closing odds**, the industry standard for sharp odds.
    - Generates a **validation report** with Brier and Log-Loss scores.
    - Produces **reliability plots** to visually assess probability calibration.
- **Reproducible & Easy to Update**: The entire workflow is scripted, from data aggregation to prediction, making it simple to retrain the model with new data.

## Project Structure

The project is organized into a clean and understandable structure:

```
football_odds_engine/
├── data/
│   ├── historical/             # Store individual season CSVs here.
│   ├── games_to_predict.csv    # Your list of upcoming games for prediction.
│   └── sample_data.csv         # The master aggregated data file (auto-generated).
├── output/
│   ├── future_predictions.csv  # Clean odds sheet for upcoming games.
│   ├── future_predictions.json # Detailed odds data for upcoming games.
│   ├── match_predictions.csv   # Predictions made on the test set (for validation).
│   ├── validation_report.md    # The model's performance report.
│   ├── reliability_plot.png    # The model's calibration chart.
│   └── trained_model.pkl       # The saved, trained model "brain".
├── src/
│   ├── dixon_coles.py          # The core Dixon-Coles model logic.
│   ├── market_calculator.py    # Functions to calculate odds from goal distributions.
│   ├── validation.py           # Calibration and validation functions.
│   └── utils.py                # Helper functions for data loading.
├── main.py                     # Main script to train the model and run validation.
├── predict_future_games.py     # Script to predict odds for upcoming games.
├── aggregate_data.py           # Script to combine all historical data.
└── requirements.txt            # List of required Python packages.
```

## Setup and Installation (From ZIP Download)

Follow these steps to get the project running on your local machine.

1.  **Unzip the Project:** Unzip `football_odds_engine.zip` to a location of your choice.

2.  **Install Python:** Ensure you have Python 3.8 or newer installed. You can download it from [python.org](https://www.python.org/downloads/).

3.  **Navigate to Project Folder:** Open your terminal or command prompt and navigate into the project's root directory.
    ```bash
    cd path/to/your/football_odds_engine
    ```

4.  **Create a Virtual Environment:** This is a best practice to keep project dependencies isolated.
    ```bash
    # For Mac/Linux
    python3 -m venv venv

    # For Windows
    python -m venv venv
    ```

5.  **Activate the Virtual Environment:**
    ```bash
    # For Mac/Linux
    source venv/bin/activate

    # For Windows
    .\venv\Scripts\activate
    ```
    Your terminal prompt should now start with `(venv)`.

6.  **Install Required Libraries:** Install all necessary packages with a single command.
    ```bash
    pip install -r requirements.txt
    ```

The environment is now fully set up and ready to use.

## Usage Workflow

This is the step-by-step process to update your data, retrain the model, and predict future games.

### Phase 1: Data Preparation

1.  **Add Historical Data:** Place all your downloaded historical season CSV files (e.g., from football-data.co.uk) into the `data/historical/` folder.
2.  **Aggregate Data:** Run the aggregation script to combine all these files into the master `sample_data.csv` file that the model uses.
    ```bash
    python aggregate_data.py
    ```

### Phase 2: Model Training & Saving

1.  **Train the Model:** Run the main script. This will train the model on your newly aggregated data, run a full validation, and save the trained model to `output/trained_model.pkl`. This is the most time-consuming step and should be done whenever you add new historical data.
    ```bash
    python main.py
    ```

### Phase 3: Predicting Future Games

1.  **Create Fixtures File:** Create a file named `data/games_to_predict.csv`. This file only needs two columns: `HomeTeam` and `AwayTeam`. List all the upcoming matches you want to generate odds for.
2.  **Run Prediction:** Execute the prediction script. This will load your saved model and generate fair odds for the matches in your fixtures file.
    ```bash
    python predict_future_games.py
    ```
3.  **Check Results:** The predictions will be saved in `output/future_predictions.csv` and `output/future_predictions.json`.

## Understanding the Output

-   **`future_predictions.csv`**: This is your primary output for betting analysis. It contains the fair decimal odds and probabilities for upcoming matches. Compare the `Odds_` columns here against bookmaker odds to find value.
-   **`validation_report.md`**: A summary of the model's historical accuracy against the sharpest market odds. Use this to understand the model's reliability.
-   **`reliability_plot.png`**: A visual check on how well-calibrated the model's probabilities are. A line close to the diagonal indicates a well-calibrated model.

## Technical Details

-   **Model:** Time-Weighted Dixon-Coles
-   **Language:** Python
-   **Core Libraries:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`
