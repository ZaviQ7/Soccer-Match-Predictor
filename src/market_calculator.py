# src/market_calculator.py

import numpy as np

def _get_fair_odds(prob):
    """Converts a probability to fair odds."""
    return 1 / prob if prob > 0 else float('inf')

def calculate_1x2(goal_matrix):
    """Calculates 1X2 (Home Win, Draw, Away Win) probabilities and fair odds."""
    home_win_prob = np.sum(np.tril(goal_matrix, -1))
    draw_prob = np.sum(np.diag(goal_matrix))
    away_win_prob = np.sum(np.triu(goal_matrix, 1))
    
    return {
        '1': {'prob': home_win_prob, 'odds': _get_fair_odds(home_win_prob)},
        'X': {'prob': draw_prob, 'odds': _get_fair_odds(draw_prob)},
        '2': {'prob': away_win_prob, 'odds': _get_fair_odds(away_win_prob)}
    }

def calculate_over_under(goal_matrix, line):
    """Calculates Over/Under probabilities and fair odds for a given total goals line."""
    total_goals_prob = np.zeros(goal_matrix.shape[0] + goal_matrix.shape[1] - 1)
    for i in range(goal_matrix.shape[0]):
        for j in range(goal_matrix.shape[1]):
            total_goals_prob[i + j] += goal_matrix[i, j]
            
    over_prob = np.sum(total_goals_prob[int(np.ceil(line + 0.01)):])
    under_prob = np.sum(total_goals_prob[:int(np.floor(line - 0.01)) + 1])
    
    # For integer lines (e.g., 2.0, 3.0), there's a push possibility
    if line == int(line):
        push_prob = total_goals_prob[int(line)]
        # The effective probability of winning is P(Over) / (1 - P(Push))
        # But for fair odds, we just report the raw probabilities
        return {
            'Over': {'prob': over_prob, 'odds': _get_fair_odds(over_prob)},
            'Under': {'prob': under_prob, 'odds': _get_fair_odds(under_prob)},
            'Push': {'prob': push_prob}
        }
    else:
        return {
            'Over': {'prob': over_prob, 'odds': _get_fair_odds(over_prob)},
            'Under': {'prob': under_prob, 'odds': _get_fair_odds(under_prob)}
        }

def calculate_asian_handicap(goal_matrix, line):
    """Calculates Asian Handicap probabilities and fair odds for the home team."""
    home_win_prob = 0
    away_win_prob = 0
    push_prob = 0

    for i in range(goal_matrix.shape[0]):
        for j in range(goal_matrix.shape[1]):
            margin = i - j
            if margin + line > 0.01:
                home_win_prob += goal_matrix[i, j]
            elif margin + line < -0.01:
                away_win_prob += goal_matrix[i, j]
            else:
                push_prob += goal_matrix[i, j]

    # Handle quarter lines (e.g., -0.25, +0.75)
    if abs(line) % 0.5 == 0.25:
        line1 = line - 0.25
        line2 = line + 0.25
        
        res1 = calculate_asian_handicap(goal_matrix, line1)
        res2 = calculate_asian_handicap(goal_matrix, line2)
        
        home_prob = (res1['Home']['prob'] + res2['Home']['prob']) / 2
        away_prob = (res1['Away']['prob'] + res2['Away']['prob']) / 2
    else:
        home_prob = home_win_prob + push_prob / 2
        away_prob = away_win_prob + push_prob / 2

    return {
        'Home': {'prob': home_prob, 'odds': _get_fair_odds(home_prob)},
        'Away': {'prob': away_prob, 'odds': _get_fair_odds(away_prob)}
    }