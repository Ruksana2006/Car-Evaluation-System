"""
File: src/app_dashboard.py
Description: Interactive CLI Dashboard for real-time Car Evaluation and Resale Valuation.
"""

import joblib
import pandas as pd

def run_dashboard():
    print("==========================================================")
    print("       🚗 SMART CAR EVALUATION & VALUATION DASHBOARD      ")
    print("==========================================================")

    # Load trained ML model
    try:
        model = joblib.load("outputs/car_evaluator_model.pkl")
    except FileNotFoundError:
        print("[ERROR] Model file not found! Please run 'python src/main_pipeline.py' first.")
        return

    # Input mappings
    price_map = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
    doors_map = {'2': 2, '3': 3, '4': 4, '5more': 5}
    persons_map = {'2': 2, '4': 4, 'more': 6}
    boot_map = {'small': 1, 'med': 2, 'big': 3}
    safety_map = {'low': 1, 'med': 2, 'high': 3}
    label_map = {0: 'UNACCEPTABLE ❌', 1: 'ACCEPTABLE ✔️', 2: 'GOOD ⭐', 3: 'VERY GOOD 🏆'}

    print("\nEnter Car Specifications for AI Evaluation:")
    buying = input("1. Buying Price Tier (low / med / high / vhigh) [default: med]: ").strip().lower() or 'med'
    maint = input("2. Maintenance Cost (low / med / high / vhigh) [default: low]: ").strip().lower() or 'low'
    doors = input("3. Number of Doors (2 / 3 / 4 / 5more) [default: 4]: ").strip().lower() or '4'
    persons = input("4. Passenger Capacity (2 / 4 / more) [default: 4]: ").strip().lower() or '4'
    boot = input("5. Luggage Boot Size (small / med / big) [default: big]: ").strip().lower() or 'big'
    safety = input("6. Safety Rating (low / med / high) [default: high]: ").strip().lower() or 'high'

    # Prepare single-row DataFrame for prediction
    features = pd.DataFrame([[
        price_map.get(buying, 2),
        price_map.get(maint, 2),
        doors_map.get(doors, 4),
        persons_map.get(persons, 4),
        boot_map.get(boot, 2),
        safety_map.get(safety, 2)
    ]], columns=['buying_price', 'maintenance_cost', 'doors', 'persons', 'luggage_boot', 'safety'])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities) * 100

    # Optional Used Car Price Estimator
    print("\n--- Market Resale Price Estimator ---")
    base_price = float(input("7. Original Showroom Price in INR [default: 1200000]: ") or 1200000)
    age = int(input("8. Age of Car in Years [default: 3]: ") or 3)
    km = int(input("9. Kilometers Driven [default: 30000]: ") or 30000)

    depreciation = min((age * 0.10) + ((km / 10000) * 0.02), 0.80)
    estimated_resale = round(base_price * (1 - depreciation), -3)

    print("\n==========================================================")
    print("                 📊 EVALUATION REPORT                     ")
    print("==========================================================")
    print(f"• AI Quality Verdict   : {label_map[prediction]}")
    print(f"• Model Confidence     : {confidence:.1f}%")
    print(f"• Est. Resale Value    : ₹{estimated_resale:,.0f} INR")
    print("==========================================================\n")

if __name__ == "__main__":
    run_dashboard()