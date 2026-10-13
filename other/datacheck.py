# Check data inconsistency for the Bot-IoT

import os
import pandas as pd
import re

DATASET_FOLDER = "data/Bot-IoT"   # change if needed

def classify_value(x):
    # NaN / missing
    if pd.isna(x):
        return "NaN"

    # Convert to string for checking
    s = str(x).strip()

    if s == "":
        return "Empty"

    # Hexadecimal
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", s):
        return "Hex"

    # Looks like a float, e.g. 1743.0
    if re.fullmatch(r"[+-]?\d+\.\d+", s):
        return "Float-like"

    # Normal integer
    if re.fullmatch(r"[+-]?\d+", s):
        return "Integer"

    # Valid-looking IPv4
    ipv4_pattern = r"^\d{1,3}(\.\d{1,3}){3}$"

    if re.fullmatch(ipv4_pattern, s):
        parts = s.split(".")

        if all(0 <= int(part) <= 255 for part in parts):
            return "IPv4"

        return "Invalid IPv4"

    # Anything else
    return "Weird"

# Scan each file under data folder
for filename in os.listdir(DATASET_FOLDER):

    if not filename.endswith(".csv"):
        continue

    filepath = os.path.join(DATASET_FOLDER, filename)

    print("\n" + "=" * 70)
    print("FILE:", filename)
    print("=" * 70)

    for chunk_num, chunk in enumerate(
        pd.read_csv(
            filepath,
            chunksize=100_000,
            low_memory=False
        )
    ):

        for column in ["saddr", "daddr"]:

            if column not in chunk.columns:
                print(f"{column}: NOT FOUND")
                continue

            print(f"\n--- {column} (chunk {chunk_num}) ---")

            # Classify every value
            types = chunk[column].apply(classify_value)

            # Print counts
            print("\nValue types:")
            print(types.value_counts())

            suspicious_types = [
                "NaN",
                "Empty",
                "Hex",
                "Float-like",
                "Invalid IPv4",
                "Weird"
            ]

            suspicious = chunk[types.isin(suspicious_types)]

            if len(suspicious) > 0:

                print("\nSuspicious values:")

                # Get unique suspicious values
                unique_values = (
                    suspicious[column]
                    .astype(str)
                    .value_counts()
                    .head(20)
                )

                print(unique_values)

            else:
                print("\nNo suspicious values found.")

        if chunk_num >= 4:
            print("\nChecked first 5 chunks.")
            break