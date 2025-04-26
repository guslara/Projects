import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# --- Load Merged Data ---
DATA_FILE = "data/cleaned/water_data_cleaned.csv"
OUTPUT_MODEL = "models/forecast_model.pkl"

# Load data
df = pd.read_csv(DATA_FILE)

# Minimal cleaning
df.columns = df.columns.str.strip()

# Encode Season as numeric
season_mapping = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
df['Season_Num'] = df['Season'].map(season_mapping)

# Features to train on
X = df[['Avg_Temp_F', 'Total_Rainfall_Inches', 'Avg_Humidity_Percent', 'Season_Num', 'Population_Density']]
y = df['Usage (GAL)']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
with open(OUTPUT_MODEL, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Model trained and saved to {OUTPUT_MODEL}")
