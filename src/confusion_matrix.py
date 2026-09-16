"""
Generates multi-class confusion matrix CSV and per-intent analysis table.
"""

import csv
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def generate_confusion_matrix():
    if not config.PREDICTIONS_PATH.exists():
        print(f"Error: {config.PREDICTIONS_PATH} does not exist. Run pipeline.py first.")
        sys.exit(1)

    with open(config.PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    y_true = [d["gold_intent"] for d in data]
    y_pred = [d["pred_intent"] for d in data]

    cm = confusion_matrix(y_true, y_pred, labels=config.INTENTS)
    cm_df = pd.DataFrame(cm, index=config.INTENTS, columns=config.INTENTS)

    # Save to CSV
    cm_df.to_csv(config.CONFUSION_MATRIX_PATH)
    print(f"Saved confusion matrix CSV to: {config.CONFUSION_MATRIX_PATH}\n")

    print("="*65)
    print("           PER-INTENT CONFUSION MATRIX (True \\ Pred)          ")
    print("="*65)
    print(cm_df.to_string())
    print("="*65)


if __name__ == '__main__':
    generate_confusion_matrix()
