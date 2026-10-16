"""
File: src/data_generator.py
Description: Generates the Car Evaluation dataset (1,728 combinations + noise/missing values)
             to demonstrate complete Data Cleaning and Preprocessing workflows.
"""

import os
import itertools
import pandas as pd
import numpy as np

def generate_car_dataset():
    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Define standard UCI Car Evaluation attributes
    buying_levels = ['vhigh', 'high', 'med', 'low']
    maint_levels = ['vhigh', 'high', 'med', 'low']
    doors_levels = ['2', '3', '4', '5more']
    persons_levels = ['2', '4', 'more']
    lug_boot_levels = ['small', 'med', 'big']
    safety_levels = ['low', 'med', 'high']

    rows = []
    # Generate all 1,728 possible car configurations
    for b, m, d, p, l, s in itertools.product(
        buying_levels, maint_levels, doors_levels, persons_levels, lug_boot_levels, safety_levels
    ):
        # Determine target evaluation class based on domain rules
        if s == 'low' or p == '2':
            target = 'unacc'
        elif b in ['vhigh', 'high'] and m in ['vhigh', 'high']:
            target = 'unacc'
        elif s == 'high' and b in ['low', 'med'] and m in ['low', 'med'] and l in ['med', 'big']:
            target = 'vgood' if (b == 'low' and l == 'big') else 'good'
        elif s == 'med' and b == 'low' and m in ['low', 'med'] and l == 'big':
            target = 'good'
        else:
            target = 'acc'

        rows.append([b, m, d, p, l, s, target])

    df = pd.DataFrame(rows, columns=[
        'buying_price', 'maintenance_cost', 'doors', 'persons', 'luggage_boot', 'safety', 'evaluation_class'
    ])

    # Inject random missing values and duplicate rows to demonstrate Data Cleaning
    np.random.seed(42)
    for col in ['buying_price', 'maintenance_cost', 'luggage_boot']:
        missing_indices = np.random.choice(df.index, size=8, replace=False)
        df.loc[missing_indices, col] = np.nan

    duplicates = df.sample(15, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    # Save raw dataset
    output_path = "data/raw_car_data.csv"
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Raw dataset generated at '{output_path}' with {len(df)} records.")

if __name__ == "__main__":
    generate_car_dataset()