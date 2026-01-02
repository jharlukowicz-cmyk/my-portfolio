"""
COMP112 Final Project
Car Insurance Modeling
Johnathan Harlukowicz
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# FILE PATHS

DATA_FILE = "/Users/johnathanharlukowicz/Downloads/insurance_claims.csv"

# 1. LOAD DATA

def load_data(filename):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
        filename (str): Path to the CSV file.

    Returns:
        DataFrame or None
    """
    if not os.path.exists(filename):
        print("ERROR: File not found:", filename)
        return None

    try:
        df = pd.read_csv(filename)
        print("Dataset loaded:", len(df), "rows.")
        return df
    except:
        print("Could not load file.")
        return None

# 2. CLEAN DATA

def clean_data(df):
    """
    Clean dataset by selecting relevant columns,
    converting numeric values, and removing missing data.

    Parameters:
        df (DataFrame): Raw dataset

    Returns:
        DataFrame: Cleaned dataset
    """

    keep_cols = [
        "age",
        "policy_deductable",
        "vehicle_claim",
        "incident_severity",
        "incident_type",
        "total_claim_amount",
        "incident_state",
        "collision_type",
        "incident_hour_of_the_day"
    ]

    cols = []
    for c in keep_cols:
        if c in df.columns:
            cols.append(c)

    df = df[cols].copy()

    numeric_cols = [
        "age", "policy_deductable",
        "vehicle_claim", "total_claim_amount",
        "incident_hour_of_the_day"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    print("Data cleaned:", len(df), "rows remain.")
    return df


# 3. SUMMARY STATISTICS

def compute_stats(df):
    """
    Compute mean, min, and max values using loops.

    Parameters:
        df (DataFrame): Cleaned dataset

    Returns:
        dict: Summary statistics
    """

    stats = {}
    columns = ["age", "total_claim_amount", "policy_deductable"]

    for col in columns:
        values = df[col]

        total = 0
        count = 0
        for v in values:
            total += v
            count += 1

        stats[col + "_mean"] = round(total / count, 2)
        stats[col + "_min"] = round(values.min(), 2)
        stats[col + "_max"] = round(values.max(), 2)

    print("\n SUMMARY STATISTICS")
    for key in stats:
        print(key + ":", stats[key])

    return stats

# 4. PLOTTING FUNCTIONS

def plot_histogram(df, col):
    """
    Plot a histogram of a numeric column.
    """
    plt.hist(df[col], bins=25)
    plt.title("Histogram of " + col)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_scatter(df, x, y):
    """
    Plot a scatter plot of two numeric variables.
    """
    plt.scatter(df[x], df[y], alpha=0.5)
    plt.title(y + " vs " + x)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()


def plot_avg_cost_by_severity(df):
    """
    Bar chart of average claim cost by severity.
    """

    sums = {}
    counts = {}

    for _, row in df.iterrows():
        sev = row["incident_severity"]
        cost = row["total_claim_amount"]

        if sev not in sums:
            sums[sev] = 0
            counts[sev] = 0

        sums[sev] += cost
        counts[sev] += 1

    severities = []
    averages = []

    for sev in sums:
        severities.append(sev)
        averages.append(sums[sev] / counts[sev])

    plt.bar(severities, averages)
    plt.title("Average Claim Cost by Severity")
    plt.xlabel("Incident Severity")
    plt.ylabel("Average Cost ($)")
    plt.grid(axis="y")
    plt.show()

# 5. PREDICTION MODEL

def predict_cost(df, age, severity):
    """
    Predict claim cost based on age and severity.

    Parameters:
        df (DataFrame)
        age (int)
        severity (str)

    Returns:
        float
    """

    matches = []

    for _, row in df.iterrows():
        if abs(row["age"] - age) <= 5 and row["incident_severity"] == severity:
            matches.append(row["total_claim_amount"])

    if len(matches) == 0:
        print("No close match found. Using overall average.")
        return df["total_claim_amount"].mean()

    total = 0
    for m in matches:
        total += m

    return total / len(matches)


def user_prediction(df):
    """
    Get user input and predict claim cost.
    Includes checks for valid age and severity.
    """

    print("\nCLAIM COST PREDICTION")

    # Get age input
    age_input = input("Enter driver age (must be 16 or older): ")

    # Check that age is numeric
    if not age_input.isdigit():
        print("Invalid input: age must be a number.")
        return

    age = int(age_input)

    # Check minimum driving age
    if age < 16:
        print("Invalid input: driver must be at least 16 years old.")
        return

    # Allowed severity options
    valid_severities = [
        "Minor Damage",
        "Moderate Damage",
        "Severe Damage",
        "Total Loss"
    ]

    print("\nSeverity options:")
    for sev in valid_severities:
        print(sev)

    severity = input("Enter incident severity exactly as shown above: ")

    # Check that severity is valid
    if severity not in valid_severities:
        print("Invalid input: severity type not recognized.")
        return

    # Make prediction
    predicted = predict_cost(df, age, severity)
    print("\nPredicted claim cost: $", round(predicted, 2))


# 6. MAIN PROGRAM

def main():
    """
    Main driver function.
    """

    df = load_data(DATA_FILE)
    if df is None:
        return

    df = clean_data(df)

    stats = compute_stats(df)

    # Graphs
    plot_histogram(df, "age")
    plot_histogram(df, "total_claim_amount")
    plot_scatter(df, "age", "total_claim_amount")
    plot_avg_cost_by_severity(df)

    # Get user input + prediction
    user_result = user_prediction(df)


# Run program
if __name__ == "__main__":
    main()
