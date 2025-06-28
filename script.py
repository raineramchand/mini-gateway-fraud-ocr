import os
import random
import string

import numpy as np
import pandas as pd

# ─── Configuration ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

N = 86_000
n_fraud = N // 801      # ≈1 fraud per 800 legits
n_legit = N - n_fraud

# ─── 1) Device IDs: 15-digit strings ────────────────────────────────────────
device_ids = [
    ''.join(random.choices(string.digits, k=15))
    for _ in range(N)
]

# ─── 2) BIN codes: split among card networks ───────────────────────────────
r = np.random.rand(N)
bin_codes = np.empty(N, dtype=int)

mask_visa = r < 0.4
bin_codes[mask_visa] = np.random.randint(400000, 500000, size=mask_visa.sum())

mask_mc = (r >= 0.4) & (r < 0.8)
bin_codes[mask_mc] = np.random.randint(510000, 560000, size=mask_mc.sum())

mask_disc = (r >= 0.8) & (r < 0.95)
bin_codes[mask_disc] = np.random.randint(600000, 700000, size=mask_disc.sum())

mask_amex = r >= 0.95
amex_idx = np.where(mask_amex)[0]
half = len(amex_idx) // 2
bin_codes[amex_idx[:half]] = np.random.randint(340000, 350000, size=half)
bin_codes[amex_idx[half:]] = np.random.randint(370000, 380000, size=len(amex_idx) - half)

# ─── 3) Geo clusters around Pakistan’s major cities ────────────────────────
clusters = np.array([
    [24.8607, 67.0011],  # Karachi
    [31.5546, 74.3572],  # Lahore
    [33.6844, 73.0479],  # Islamabad
    [34.0151, 71.5249],  # Peshawar
    [30.1798, 66.9750],  # Quetta
])
weights = [0.3, 0.3, 0.2, 0.1, 0.1]
choices = np.random.choice(len(clusters), size=N, p=weights)
geo_lat = clusters[choices, 0] + np.random.normal(0, 0.05, size=N)
geo_long = clusters[choices, 1] + np.random.normal(0, 0.05, size=N)

# ─── 4) Amounts: log-normal (fraud skews larger) ──────────────────────────
legit_amounts = np.random.lognormal(mean=3, sigma=0.5, size=n_legit)
fraud_amounts = np.random.lognormal(mean=4, sigma=1.0, size=n_fraud)
amounts = np.concatenate([legit_amounts, fraud_amounts])
amounts = np.round(amounts, 2)

# ─── 5) Labels & assemble ─────────────────────────────────────────────────
labels = np.concatenate([
    np.zeros(n_legit, dtype=int),
    np.ones(n_fraud, dtype=int),
])

df = pd.DataFrame({
    'device_id': device_ids,
    'bin':        bin_codes,
    'geo_lat':    np.round(geo_lat, 6),
    'geo_long':   np.round(geo_long, 6),
    'amount':     amounts,
    'is_fraud':   labels
})

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ─── 6) Save to CSV ────────────────────────────────────────────────────────
out_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(out_dir, 'synthetic_micro_merchant_transactions.csv')
df.to_csv(csv_path, index=False)

# ─── 7) Quick sanity check ─────────────────────────────────────────────────
print("Sample rows:")
print(df.head().to_string(index=False))
size_mb = os.path.getsize(csv_path) / (1024 * 1024)
print(f"\nSaved {N:,} rows → {csv_path} ({size_mb:.2f} MB)")
