import pandas as pd
import os

# --- Setup Paths ---
raw_folder = "data/raw/"
output_folder = "data/cleaned/"
output_file = os.path.join(output_folder, "water_data_cleaned.csv")

os.makedirs(output_folder, exist_ok=True)

# --- List of Original Raw Files ---
files_to_merge = [
    "Residential_consump_0_row_1000000_b3781101-5e07-42ac-a471-147187e75d6e.csv",
    "Residential_consump_1_row_1000000_b6ce8f38-8767-439e-8b8b-cd0d878a084a.csv",
    "Residential_consump_2_row_1000000_cca56f02-5d5c-4303-a368-ecab6f893d94.csv"
]

# --- Load and Merge ---
df_list = []

for file in files_to_merge:
    file_path = os.path.join(raw_folder, file)
    print(f"Loading {file_path}")
    df = pd.read_csv(file_path)
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)
print(f"Merged {len(df_list)} files. Total rows: {merged_df.shape[0]}")

# --- Clean ---
# Fix ZIP to 5-digit string
merged_df['ZIP'] = merged_df['ZIP'].astype(str).str.zfill(5)

# Parse Billing Date
merged_df['Billing Date'] = pd.to_datetime(merged_df['Billing Date'], errors='coerce')

# Drop rows missing critical fields
merged_df = merged_df.dropna(subset=['Billing Date', 'Usage (GAL)', 'ZIP'])

# Optional: Filter out crazy outlier values if needed
# merged_df = merged_df[merged_df['Usage (GAL)'] < 500000]

# --- Save Cleaned Version ---
merged_df.to_csv(output_file, index=False)

print(f"Saved cleaned data to {output_file}")
