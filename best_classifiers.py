import os
import glob
import pandas as pd
import numpy as np

def process_csv(file_path):
    """Processes a CSV file and saves the Pareto-optimal classifiers."""

    if "Results" in os.path.normpath(file_path).split(os.sep) or file_path.endswith("_pareto_optimal.csv"):
        print(f"Skipping: {file_path} (either inside 'Results' directory or already processed)")
        return

    # Load CSV file
    df = pd.read_csv(file_path)

    df = df.dropna()

    metrics = ["Accuracy", "Precision", "Recall", "F1", "MSE", "FP", "FN"]
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    maximize = ["Accuracy", "Precision", "Recall", "F1"]  # Metrics to maximize

    # Create cost matrix with flipped minimization metrics
    cost_matrix = df[maximize].copy()


    def is_pareto_efficient(costs):
        """Returns a boolean array indicating which points are Pareto optimal."""
        num_points = costs.shape[0]
        is_efficient = np.ones(num_points, dtype=bool)
        for i in range(num_points):
            if is_efficient[i]:
                # Compare with all other points
                is_efficient[i] = not np.any(np.all(costs > costs[i], axis=1))
        return is_efficient

    pareto_mask = is_pareto_efficient(cost_matrix.to_numpy())
    pareto_df = df[pareto_mask]

    pareto_df = pareto_df.sort_values(by="Accuracy", ascending=False)

    top_classifiers = pareto_df.head(6)

    print(f"\nTop Pareto-Optimal Classifiers for {file_path}:")
    print(top_classifiers)

    # Define output file path
    output_file = os.path.splitext(file_path)[0] + "_pareto_optimal.csv"

    # Save results to CSV
    top_classifiers.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")

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
