import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print("Step 1: Loading CSV...")
df = pd.read_csv("online_retail.csv", encoding="latin1")

print("Step 2: Cleaning data...")
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df["Description"] = df["Description"].str.strip().str.upper()
df["CustomerID"] = df["CustomerID"].astype(int)
print(f"  Clean rows: {len(df):,}")

print("Step 3: Building similarity matrix (1-2 min)...")
pivot = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    fill_value=0
)
cos_sim = cosine_similarity(pivot.T)
cos_df = pd.DataFrame(cos_sim, index=pivot.columns, columns=pivot.columns)

print("Step 4: Saving...")
cos_df.to_pickle("models/cosine_sim.pkl")
print("DONE! cosine_sim.pkl saved to models/")
