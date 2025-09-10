# src/validation.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

def plot_reliability_diagram(y_true, y_prob, n_bins=10, title='Reliability Plot'):
    """
    Generates and saves a reliability plot.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, 's-', label='Model')
    plt.xlabel('Mean predicted probability (fraction of positives)')
    plt.ylabel('Fraction of positives (true probability)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('output/reliability_plot.png')
    print("Reliability plot saved to output/reliability_plot.png")
    plt.close()

def get_calibration_scores(y_true, y_prob):
    """
    Calculates Brier and Log-Loss scores.
    """
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    return {'brier_score': brier, 'log_loss': ll}

def calibrate_probabilities(probs_to_calibrate, reference_probs, reference_outcomes):
    """
    Calibrates probabilities using Isotonic Regression.
    
    Args:
        probs_to_calibrate (array): The probabilities to be calibrated (e.g., from test set).
        reference_probs (array): The probabilities to train the calibrator on (e.g., from train set).
        reference_outcomes (array): The true outcomes for the reference set.
        
    Returns:
        Calibrated probabilities.
    """
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(reference_probs, reference_outcomes)
    calibrated_probs = iso_reg.predict(probs_to_calibrate)
    return calibrated_probs