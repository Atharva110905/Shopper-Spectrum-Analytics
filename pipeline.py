"""
utils/pipeline.py
Full ETL + ML training pipeline for Shopper Spectrum.
Run this script once (or whenever dataset changes) to regenerate all data/ and models/ artifacts.

Usage:
    python utils/pipeline.py --data path/to/online_retail.csv
"""

import argparse
import os
import json
import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA_DIR   = BASE / "data"
MODELS_DIR = BASE / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load CSV and apply all cleaning steps."""
    print(f"[1/5] Loading dataset: {filepath}")
    df = pd.read_csv(filepath, encoding="latin1")
    raw_rows = len(df)

    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.drop_duplicates()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"]  = df["CustomerID"].astype(int)
    df["Description"] = df["Description"].str.strip().str.upper()
    df["Month"]       = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Hour"]        = df["InvoiceDate"].dt.hour
    df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()

    clean_rows = len(df)
    print(f"    Raw: {raw_rows:,} → Clean: {clean_rows:,} rows "
          f"({raw_rows - clean_rows:,} removed)")
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM values per customer."""
    print("[2/5] Computing RFM values …")
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency  =("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo",   "nunique"),
        Monetary =("TotalPrice",  "sum"),
    ).reset_index()
    print(f"    {len(rfm):,} customers with RFM scores")
    return rfm


def cluster_customers(rfm: pd.DataFrame, n_clusters: int = 4):
    """Scale RFM, run KMeans, label segments, return updated rfm + artifacts."""
    print("[3/5] Clustering customers …")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    inertias, silhouettes = [], []
    for k in range(2, 9):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(rfm_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(rfm_scaled, lbl)))
        print(f"    k={k}  inertia={km.inertia_:,.0f}  silhouette={silhouette_score(rfm_scaled, lbl):.3f}")

    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = km_final.fit_predict(rfm_scaled)

    profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    profile["Score"] = (
        -profile["Recency"]
        + profile["Frequency"] * 10
        + profile["Monetary"] / 1000
    )
    rank_order = profile["Score"].rank(ascending=False)
    labels_map: dict[int, str] = {}
    for idx, rank in rank_order.items():
        if rank == 1:   labels_map[idx] = "High-Value"
        elif rank == 2: labels_map[idx] = "Regular"
        elif rank == 3: labels_map[idx] = "Occasional"
        else:           labels_map[idx] = "At-Risk"

    rfm["Segment"] = rfm["Cluster"].map(labels_map)
    print("    Segment distribution:")
    for seg, cnt in rfm["Segment"].value_counts().items():
        print(f"      {seg}: {cnt:,}")

    return rfm, scaler, km_final, labels_map, inertias, silhouettes


def build_recommendation(df: pd.DataFrame):
    """Build item-item cosine similarity matrix."""
    print("[4/5] Building product recommendation matrix …")
    pivot   = df.pivot_table(index="CustomerID", columns="Description",
                             values="Quantity", fill_value=0)
    cos_sim = cosine_similarity(pivot.T)
    cos_df  = pd.DataFrame(cos_sim, index=pivot.columns, columns=pivot.columns)
    print(f"    Similarity matrix: {cos_df.shape}")
    return cos_df, list(pivot.columns)


def save_all(df, rfm, cos_df, products, scaler, km_final,
             labels_map, inertias, silhouettes):
    """Save all artifacts to data/ and models/."""
    print("[5/5] Saving artifacts …")

    # ── RFM ──
    rfm.to_csv(DATA_DIR / "rfm.csv", index=False)

    # ── Elbow ──
    json.dump(
        {"k": list(range(2, 9)), "inertia": inertias, "silhouette": silhouettes},
        open(DATA_DIR / "elbow.json", "w"),
    )

    # ── Cluster labels ──
    json.dump(
        {str(k): v for k, v in labels_map.items()},
        open(DATA_DIR / "cluster_labels.json", "w"),
    )

    # ── Monthly ──
    monthly = df.groupby("Month").agg(
        Revenue  =("TotalPrice", "sum"),
        Orders   =("InvoiceNo",  "nunique"),
        Customers=("CustomerID", "nunique"),
    ).reset_index()
    monthly.to_csv(DATA_DIR / "monthly.csv", index=False)

    # ── Country ──
    country = (
        df.groupby("Country")
        .agg(Revenue=("TotalPrice", "sum"), Orders=("InvoiceNo", "nunique"))
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    country.to_csv(DATA_DIR / "country.csv", index=False)

    # ── Top products ──
    top_p = (
        df.groupby("Description")
        .agg(Quantity=("Quantity", "sum"), Revenue=("TotalPrice", "sum"))
        .sort_values("Revenue", ascending=False)
        .head(20)
        .reset_index()
    )
    top_p.to_csv(DATA_DIR / "top_products.csv", index=False)

    # ── Heatmap ──
    heatmap = df.groupby(["DayOfWeek", "Hour"])["TotalPrice"].sum().reset_index()
    heatmap.to_csv(DATA_DIR / "heatmap.csv", index=False)

    # ── KPIs ──
    kpis = {
        "total_revenue"  : round(float(df["TotalPrice"].sum()), 2),
        "total_orders"   : int(df["InvoiceNo"].nunique()),
        "total_customers": int(df["CustomerID"].nunique()),
        "total_products" : int(df["Description"].nunique()),
        "avg_order_value": round(float(df.groupby("InvoiceNo")["TotalPrice"].sum().mean()), 2),
        "top_country"    : str(df.groupby("Country")["TotalPrice"].sum().idxmax()),
        "date_min"       : str(df["InvoiceDate"].min().date()),
        "date_max"       : str(df["InvoiceDate"].max().date()),
    }
    json.dump(kpis, open(DATA_DIR / "kpis.json", "w"))

    # ── Products list ──
    json.dump(products, open(DATA_DIR / "products.json", "w"))

    # ── Models ──
    pickle.dump(scaler,   open(MODELS_DIR / "scaler.pkl", "wb"))
    pickle.dump(km_final, open(MODELS_DIR / "kmeans.pkl", "wb"))
    cos_df.to_pickle(MODELS_DIR / "cosine_sim.pkl")

    print("    All artifacts saved.")
    print(f"\n✅ Pipeline complete! Run:  streamlit run {BASE}/app.py")


def main():
    parser = argparse.ArgumentParser(description="Shopper Spectrum ETL + ML Pipeline")
    parser.add_argument(
        "--data",
        default="/mnt/user-data/uploads/online_retail.csv",
        help="Path to the online_retail.csv file",
    )
    parser.add_argument("--clusters", type=int, default=4,
                        help="Number of KMeans clusters (default: 4)")
    args = parser.parse_args()

    df                             = load_and_clean(args.data)
    rfm                            = compute_rfm(df)
    rfm, scaler, km, lmap, ine, sil = cluster_customers(rfm, args.clusters)
    cos_df, products               = build_recommendation(df)
    save_all(df, rfm, cos_df, products, scaler, km, lmap, ine, sil)


if __name__ == "__main__":
    main()
