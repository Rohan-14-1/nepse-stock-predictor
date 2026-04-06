import pandas as pd
import numpy as np
from nepse_analysis import fit_ml_classifiers
from nepse_engine import generate_synthetic_data

def test_variation():
    tickers = ["NABIL", "SOHL", "ADBL", "HFIN", "SANIMA"]
    for t in tickers:
        df = generate_synthetic_data(t)
        # Manually inject a "Big Move" today to see if AI reacts
        last_idx = df.index[-1]
        df.loc[last_idx, 'Close'] *= 1.05 # Fake 5% spike
        from nepse_engine import compute_features
        df_feat = compute_features(df)
        
        result = fit_ml_classifiers(df_feat)
        print(f"Ticker: {t}")
        print(f"  Best Model: {result['best_model']}")
        print(f"  Profit Prob: {result['profit_probability']}%")
        print(f"  Prediction: {result['next_day_prediction']}")

if __name__ == "__main__":
    test_variation()
