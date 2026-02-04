#!/usr/bin/env python3
# ============================================================
# UIDAI DATA HACKATHON 2026
# End-to-End Analytical Pipeline
# ============================================================
# Author: <Your Name>
# Description:
# Explainable anomaly detection, persistence analysis,
# and governance-ready insights from Aadhaar data
# ============================================================

import os
import sys
import logging
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CELL 1: Environment setup & logging
# ============================================================

LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("UIDAI_PIPELINE")

logger.info("Starting UIDAI Aadhaar Data Processing Pipeline")

USE_GPU = False
try:
    import cudf
    import cupy as cp
    pd = cudf
    np = cp
    USE_GPU = True
    logger.info("GPU detected → Using cuDF / CuPy")
except Exception:
    import pandas as pd
    import numpy as np
    logger.info("GPU not available → Using Pandas / NumPy")

SEED = 42
np.random.seed(SEED)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "./data"
REF_DIR = "./data_reference"
OUTPUT_DIR = "./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CELL 2: Load raw CSV shards
# ============================================================

def load_group(file_list, label):
    logger.info(f"Loading {label} data")
    dfs = []
    for f in file_list:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        dfs.append(df)
        logger.info(f"Loaded {os.path.basename(f)} | rows={len(df):,}")
    return pd.concat(dfs, ignore_index=True)

bio_files = [
    f"{DATA_DIR}/api_data_aadhar_biometric_0_500000.csv",
    f"{DATA_DIR}/api_data_aadhar_biometric_500000_1000000.csv",
    f"{DATA_DIR}/api_data_aadhar_biometric_1000000_1500000.csv",
    f"{DATA_DIR}/api_data_aadhar_biometric_1500000_1861108.csv",
]

demo_files = [
    f"{DATA_DIR}/api_data_aadhar_demographic_0_500000.csv",
    f"{DATA_DIR}/api_data_aadhar_demographic_500000_1000000.csv",
    f"{DATA_DIR}/api_data_aadhar_demographic_1000000_1500000.csv",
    f"{DATA_DIR}/api_data_aadhar_demographic_1500000_2000000.csv",
    f"{DATA_DIR}/api_data_aadhar_demographic_2000000_2071700.csv",
]

enrol_files = [
    f"{DATA_DIR}/api_data_aadhar_enrolment_0_500000.csv",
    f"{DATA_DIR}/api_data_aadhar_enrolment_500000_1000000.csv",
    f"{DATA_DIR}/api_data_aadhar_enrolment_1000000_1006029.csv",
]

df_bio = load_group(bio_files, "Biometric")
df_demo = load_group(demo_files, "Demographic")
df_enrol = load_group(enrol_files, "Enrolment")

# ============================================================
# CELL 3: Date parsing
# ============================================================

def parse_dates(df):
    df["date"] = (
        df["date"]
        .astype(str)
        .str.replace("/", "-")
        .str.replace(".", "-")
    )
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    return df

df_bio = parse_dates(df_bio)
df_demo = parse_dates(df_demo)
df_enrol = parse_dates(df_enrol)

# ============================================================
# CELL 4–7: Geographic normalization
# ============================================================

def clean_geo(s):
    return (
        s.astype(str)
        .str.upper()
        .str.strip()
        .str.replace("&", "AND", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

for df in [df_bio, df_demo, df_enrol]:
    df["state"] = clean_geo(df["state"])
    df["district_clean"] = clean_geo(df["district"])
    df["pincode"] = df["pincode"].astype(str).str.zfill(6)

# ============================================================
# CELL 9: Deduplication
# ============================================================

def dedupe(df, name):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"{name}: removed {before-after:,} exact duplicates")
    return df

df_bio = dedupe(df_bio, "Biometric")
df_demo = dedupe(df_demo, "Demographic")
df_enrol = dedupe(df_enrol, "Enrolment")

# ============================================================
# CELL 10: Volume metrics
# ============================================================

df_bio["total_bio"] = df_bio["bio_age_5_17"] + df_bio["bio_age_17_"]
df_demo["total_demo"] = df_demo["demo_age_5_17"] + df_demo["demo_age_17_"]
df_enrol["total_enrol"] = (
    df_enrol["age_0_5"] +
    df_enrol["age_5_17"] +
    df_enrol["age_18_greater"]
)

# ============================================================
# CELL 11–12: Aggregation & merge
# ============================================================

KEYS = ["date", "state", "district_clean", "pincode"]

bio_agg = df_bio.groupby(KEYS)["total_bio"].sum().reset_index()
demo_agg = df_demo.groupby(KEYS)["total_demo"].sum().reset_index()
enrol_agg = df_enrol.groupby(KEYS)["total_enrol"].sum().reset_index()

master = (
    bio_agg
    .merge(demo_agg, on=KEYS, how="outer")
    .merge(enrol_agg, on=KEYS, how="outer")
    .fillna(0)
)

# ============================================================
# CELL 14: Rolling baselines & shocks
# ============================================================

master = master.sort_values(["state", "district_clean", "date"])

WINDOW = 7
EPS = 1.0

master["bio_baseline"] = (
    master.groupby(["state", "district_clean"])["total_bio"]
    .transform(lambda x: x.rolling(WINDOW, min_periods=5).mean())
)

master["enrol_baseline"] = (
    master.groupby(["state", "district_clean"])["total_enrol"]
    .transform(lambda x: x.rolling(WINDOW, min_periods=5).mean())
)

master["bio_dev"] = (master["total_bio"] + EPS) / (master["bio_baseline"] + EPS)
master["enrol_dev"] = (master["total_enrol"] + EPS) / (master["enrol_baseline"] + EPS)

master["structural_shock"] = (
    (master["bio_dev"] >= 3.0) &
    (master["enrol_dev"] <= 0.25)
).astype(int)

# ============================================================
# CELL 16: Change-point detection (CPU)
# ============================================================

import ruptures as rpt

cp_results = []

cpu_df = master.to_pandas() if USE_GPU else master.copy()

for (state, dist), g in cpu_df.groupby(["state", "district_clean"]):
    if len(g) < 30:
        continue
    series = g["total_bio"].values
    algo = rpt.Pelt(model="rbf").fit(series)
    cps = algo.predict(pen=10)[:-1]
    for i in cps:
        cp_results.append({
            "state": state,
            "district_clean": dist,
            "date": g.iloc[i]["date"],
            "change_point": 1
        })

cpd = pd.DataFrame(cp_results)

# ============================================================
# CELL 19: Socio-economic enrichment (context only)
# ============================================================

census = pd.read_csv(f"{REF_DIR}/census_district_2011.csv")
census["state"] = clean_geo(census["state"])
census["district_clean"] = clean_geo(census["district"])

cpu_df = cpu_df.merge(
    census[["state", "district_clean", "population", "literacy_rate"]],
    on=["state", "district_clean"],
    how="left"
)

cpu_df["bio_per_10k_pop"] = cpu_df["total_bio"] * 10000 / (cpu_df["population"] + 1)

# ============================================================
# CELL 20: Composite anomaly score
# ============================================================

cpu_df = cpu_df.merge(
    cpd, on=["state", "district_clean", "date"], how="left"
)
cpu_df["change_point"] = cpu_df["change_point"].fillna(0)

cpu_df["raw_score"] = (
    0.4 * cpu_df["structural_shock"] +
    0.3 * np.log1p(cpu_df["bio_dev"]) +
    0.3 * np.log1p(1 / (cpu_df["enrol_dev"] + 1e-3))
)

cpu_df["anomaly_percentile"] = cpu_df["raw_score"].rank(pct=True)

def severity(p):
    if p >= 0.99:
        return "CRITICAL"
    if p >= 0.95:
        return "HIGH"
    if p >= 0.90:
        return "MEDIUM"
    return "NORMAL"

cpu_df["anomaly_severity"] = cpu_df["anomaly_percentile"].apply(severity)

# ============================================================
# CELL 21: Persistence analysis
# ============================================================

cpu_df = cpu_df.sort_values(["state", "district_clean", "date"])

cpu_df["flag"] = cpu_df["anomaly_severity"].isin(["HIGH", "CRITICAL"]).astype(int)
cpu_df["segment"] = cpu_df.groupby(
    ["state", "district_clean"]
)["flag"].diff().fillna(0).ne(0).cumsum()

cpu_df["days_persistent"] = cpu_df.groupby(
    ["state", "district_clean", "segment"]
)["flag"].transform("sum")

def persistence_label(d):
    if d >= 21:
        return "CHRONIC"
    if d >= 7:
        return "PERSISTENT"
    if d > 0:
        return "NEW"
    return "NONE"

cpu_df["persistence_category"] = cpu_df["days_persistent"].apply(persistence_label)

# ============================================================
# SAVE OUTPUTS
# ============================================================

cpu_df.to_csv(f"{OUTPUT_DIR}/district_day_final_analysis.csv", index=False)

logger.info("Pipeline completed successfully")
