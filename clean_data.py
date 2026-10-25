"""
BigMart Sales Prediction — Phase 2: Data Cleaning
Fixes: inconsistent Item_Fat_Content labels, missing Item_Weight,
       missing Outlet_Size, implausible Item_Visibility == 0.

Golden rule: every statistic (mean, mode) is learned from TRAIN ONLY,
then applied identically to test. Nothing is fit on test.
"""
import pandas as pd
import numpy as np

RAW = "data/raw"
OUT = "data/processed"

train = pd.read_csv(f"{RAW}/train_v9rqX0R.csv")
test = pd.read_csv(f"{RAW}/test_AbJTz2l.csv")

print("=" * 70)
print("BEFORE CLEANING")
print("=" * 70)
print(f"train shape: {train.shape}   test shape: {test.shape}")
print("\nMissing values (train):\n", train.isnull().sum()[train.isnull().sum() > 0])
print("\nMissing values (test):\n", test.isnull().sum()[test.isnull().sum() > 0])
print("\nItem_Fat_Content raw labels:\n", train["Item_Fat_Content"].value_counts())
print(f"\nItem_Visibility == 0 rows -> train: {(train['Item_Visibility']==0).sum()}, "
      f"test: {(test['Item_Visibility']==0).sum()}")

# ---------------------------------------------------------------------
# 2.1  Fix Item_Fat_Content labels: 5 messy labels -> 2 clean ones
# ---------------------------------------------------------------------
fat_map = {
    "LF": "Low Fat", "low fat": "Low Fat", "Low Fat": "Low Fat",
    "reg": "Regular", "Regular": "Regular",
}
train["Item_Fat_Content"] = train["Item_Fat_Content"].map(fat_map)
test["Item_Fat_Content"] = test["Item_Fat_Content"].map(fat_map)

# ---------------------------------------------------------------------
# 2.2  Impute Item_Weight via per-product (Item_Identifier) mean,
#      learned on train only; fallback to train's overall mean.
# ---------------------------------------------------------------------
item_weight_map = train.groupby("Item_Identifier")["Item_Weight"].mean()
overall_weight_mean = train["Item_Weight"].mean()

for df in (train, test):
    df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Identifier"].map(item_weight_map))
train["Item_Weight"] = train["Item_Weight"].fillna(overall_weight_mean)
test["Item_Weight"] = test["Item_Weight"].fillna(overall_weight_mean)

# ---------------------------------------------------------------------
# 2.3  Impute Outlet_Size via mode of Outlet_Size within Outlet_Type,
#      learned on train only.
# ---------------------------------------------------------------------
mode_by_type = train.groupby("Outlet_Type")["Outlet_Size"].agg(lambda x: x.mode().iloc[0])

for df in (train, test):
    df["Outlet_Size"] = df.apply(
        lambda r: mode_by_type[r["Outlet_Type"]] if pd.isnull(r["Outlet_Size"]) else r["Outlet_Size"],
        axis=1,
    )

# ---------------------------------------------------------------------
# 2.4  Treat Item_Visibility == 0 as missing, impute via per-product
#      mean visibility, learned on train only; fallback to train mean.
# ---------------------------------------------------------------------
for df in (train, test):
    df["Item_Visibility"] = df["Item_Visibility"].replace(0, np.nan)

vis_map = train.groupby("Item_Identifier")["Item_Visibility"].mean()
overall_vis_mean = train["Item_Visibility"].mean()

for df in (train, test):
    df["Item_Visibility"] = df["Item_Visibility"].fillna(df["Item_Identifier"].map(vis_map))
train["Item_Visibility"] = train["Item_Visibility"].fillna(overall_vis_mean)
test["Item_Visibility"] = test["Item_Visibility"].fillna(overall_vis_mean)

# ---------------------------------------------------------------------
# 2.5  Sanity checks
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("AFTER CLEANING")
print("=" * 70)
print("Missing values (train):\n", train.isnull().sum().sum(), "total missing cells")
print("Missing values (test):\n", test.isnull().sum().sum(), "total missing cells")
print("\nItem_Fat_Content clean labels:\n", train["Item_Fat_Content"].value_counts())
print(f"\nItem_Visibility == 0 rows remaining -> train: {(train['Item_Visibility']==0).sum()}, "
      f"test: {(test['Item_Visibility']==0).sum()}")
print(f"Item_Visibility min (train): {train['Item_Visibility'].min():.6f}")

assert train.drop(columns=["Item_Outlet_Sales"]).isnull().sum().sum() == 0, "Unexpected NaNs remain in train!"
assert test.isnull().sum().sum() == 0, "Unexpected NaNs remain in test!"
assert set(train["Item_Fat_Content"].unique()) == {"Low Fat", "Regular"}, "Fat content not fully cleaned!"

train.to_csv(f"{OUT}/train_clean.csv", index=False)
test.to_csv(f"{OUT}/test_clean.csv", index=False)
print(f"\nSaved: {OUT}/train_clean.csv  ({train.shape})")
print(f"Saved: {OUT}/test_clean.csv   ({test.shape})")
print("\nAll checks passed. Data cleaning (Phase 2) complete.")
