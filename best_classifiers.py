import os
import glob
import pandas as pd
import numpy as np

def process_csv(file_path):
    """Processes a CSV file and saves the Pareto-optimal classifiers."""

    if "Results" in os.path.normpath(file_path).split(os.sep) or file_path.endswith("_pareto_optimal.csv"):
        print(f"Skipping: {file_path} (either inside 'Results' directory or already processed)")
        return

    df = pd.read_csv(file_path)

    required_columns = ["TP", "TN", "FP", "FN", "MSE", "Accuracy", "Precision", "Recall", "F1"]

    df[required_columns] = df[required_columns].apply(pd.to_numeric, errors="coerce")

    df = df[
        (df["Precision"] > 0) &
        (df["Recall"] > 0) &
        (df["F1"] > 0)
    ]

    if df.empty:
        print(f"Skipping {file_path}: No valid data after filtering missing or zero metrics.")
        return

    maximize = ["TP", "TN", "Accuracy", "Precision", "Recall", "F1"]
    minimize = ["FP", "FN", "MSE"]

    df[minimize] = -df[minimize]

    cost_matrix = df[maximize + minimize].to_numpy()



    # Compute Pareto-optimal classifiers
    pareto_mask = is_pareto_efficient(cost_matrix)
    pareto_df = df[pareto_mask].copy()

    # Restore original values for minimization metrics
    pareto_df[minimize] = -pareto_df[minimize]

    # Sort based on Accuracy (or another key metric)
    pareto_df = pareto_df.sort_values(by="Accuracy", ascending=False)

    # Ensure at least 6 results are returned
    if len(pareto_df) < 7:
        # Get additional classifiers to fill up to 6, sorted by Accuracy
        remaining_df = df[~df.index.isin(pareto_df.index)].sort_values(by="Accuracy", ascending=False)
        pareto_df = pd.concat([pareto_df, remaining_df.head(7 - len(pareto_df))])

    # Select top 6 classifiers
    top_classifiers = pareto_df.head(7)

    print(f"\nTop 6 Classifiers for {file_path}:")
    print(top_classifiers)

    # Define output file path
    output_file = os.path.splitext(file_path)[0] + "_pareto_optimal.csv"

    # Save results to CSV
    top_classifiers.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")
    
def is_pareto_efficient(costs):
    """Returns a boolean array indicating which points are Pareto optimal using NumPy broadcasting."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[i] = not np.any(np.all(costs > c, axis=1))  # Strict dominance check
    return is_efficient

# Set base directory correctly
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()  # Fallback for interactive environments

csv_files = [
    f for f in glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    if "Results" not in os.path.normpath(f).split(os.sep) and not f.endswith("_pareto_optimal.csv")
]

for csv_file in csv_files:
    process_csv(csv_file)
