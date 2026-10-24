"""
BigMart Sales Prediction - final version (simple, step by step)
================================================================

THE IDEA IN ONE LINE
    Sales = Item_MRP x (units sold).
    Units sold depend mostly on the OUTLET, and a little on the ITEM NUMBER in Item_Identifier.

THE TWO MODELS
    Stage 1: for every outlet, fit a straight line   units = a + b x MRP      (Ridge regression)
    Stage 2: look at how wrong Stage 1 is for each item number and correct it (gradient boosting)

    final sales prediction = Stage 1 units x Stage 2 correction x MRP

STEPS IN THIS FILE
    Step 1  Load the data
    Step 2  Look at the data (checks and a few simple facts)
    Step 3  Prepare the target and the features
    Step 4  Build the models (Stage 1 and Stage 2)
    Step 5  Test the models with cross-validation and compare with simple baselines
    Step 6  Train on all training data, predict the test set and save the CSV
    Step 7  Check the saved file

HOW TO RUN:  python bigmart_final.py   (the train and test CSV files must be in the same folder)
OUTPUT:      bigmart_predictions.csv
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')       # keep the printed output clean

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------------------
# Settings (all the numbers you may want to change are here)
# ---------------------------------------------------------------------
TRAIN_PATH = 'train_v9rqX0R.csv'
TEST_PATH = 'test_AbJTz2l.csv'
OUT_PATH = 'bigmart_predictions.csv'

RIDGE_ALPHA = 1.0                 # strength of the Ridge penalty (Stage 1)
MIN_UNITS = 0.3                   # Stage 1 units are never allowed below this (avoids dividing by ~0)

STAGE2_SETTINGS = dict(           # Stage 2 gradient boosting, kept small on purpose
    max_iter=200,                 # number of small trees
    learning_rate=0.03,           # how much each tree adds
    max_depth=2,                  # each tree is very shallow
    min_samples_leaf=300,         # every leaf must contain many rows, so it cannot memorise noise
    l2_regularization=20.0,       # extra penalty against over-fitting
    random_state=0,
)

N_FOLDS = 5                       # cross-validation: 5 parts
CV_SEEDS = [42, 1, 2, 3, 4]       # 5 different random splits, to see that the result is stable


# =====================================================================
# STEP 1: LOAD THE DATA
# =====================================================================
print('=' * 60)
print('STEP 1: Load the data')
print('=' * 60)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print('Train rows and columns:', train.shape)
print('Test rows and columns :', test.shape)

# The final file needs these two id columns, so keep them now
test_ids = test[['Item_Identifier', 'Outlet_Identifier']].copy()


# =====================================================================
# STEP 2: LOOK AT THE DATA
# =====================================================================
print()
print('=' * 60)
print('STEP 2: Look at the data')
print('=' * 60)

# 2a. Missing values. (Item_Weight and Outlet_Size have gaps, but we do not use those columns.)
print('Missing values in the training data:')
missing = train.isnull().sum()
print(missing[missing > 0].to_string())

# 2b. Sales are very different from outlet to outlet, so the outlet must be in the model.
print()
print('Average sales per outlet:')
print(train.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean().round(0).to_string())

# 2c. Outlet_Size, Outlet_Type, Outlet_Location_Type and Outlet_Establishment_Year never change
#     inside one outlet, so Outlet_Identifier already contains all of that information.
#     That is why we only use Outlet_Identifier.
#     The list of outlets (in a fixed order) is used later to build the features.
OUTLETS = sorted(train['Outlet_Identifier'].unique())
print()
print('Number of outlets:', len(OUTLETS))


# =====================================================================
# STEP 3: PREPARE THE TARGET AND THE FEATURES
# =====================================================================
print()
print('=' * 60)
print('STEP 3: Prepare the target and the features')
print('=' * 60)


def get_units(df):
    """Target: units sold = sales / MRP.
    We predict units, and multiply by MRP at the end to get sales back."""
    return df['Item_Outlet_Sales'].values / df['Item_MRP'].values


def outlet_mrp_features(df):
    """Features for Stage 1.

    For every outlet we make two columns:
        <outlet>          1 if the row belongs to this outlet, otherwise 0     -> gives the outlet its own intercept
        <outlet>_x_mrp    the scaled MRP if the row belongs to this outlet,
                          otherwise 0                                           -> gives the outlet its own slope
    A linear model on these columns learns one line per outlet:
        units = intercept_outlet + slope_outlet x MRP
    """
    # Scale MRP so that the numbers are small and close to 0 (this helps Ridge)
    mrp_scaled = (df['Item_MRP'] - 140) / 60

    features = pd.DataFrame(index=df.index)

    for outlet in OUTLETS:
        outlet_flags = (df['Outlet_Identifier'] == outlet).astype(float)
        features[outlet] = outlet_flags
        features[outlet + '_x_mrp'] = outlet_flags * mrp_scaled

    return features.values


def item_number_feature(df):
    """Feature for Stage 2: the number inside Item_Identifier.
    Example: 'FDA15' -> 15, 'DRC01' -> 1.
    This number has a small but repeatable effect on how much an item sells."""
    numbers = df['Item_Identifier'].str[3:].astype(int)
    return numbers.values.reshape(-1, 1)         # the models expect a 2-D table


# Quick demonstration on the first rows, so the features are easy to understand
example = train.head(3)
print('Example Item_Identifiers      :', list(example['Item_Identifier']))
print('Their item numbers (Stage 2)  :', [int(n) for n in item_number_feature(example).ravel()])
print('Stage 1 feature table shape   :', outlet_mrp_features(train).shape)


# =====================================================================
# STEP 4: BUILD THE MODELS
# =====================================================================
def fit_stage1(train_df):
    """Stage 1: Ridge regression, one line (intercept + MRP slope) per outlet."""
    units = get_units(train_df)

    stage1 = Ridge(alpha=RIDGE_ALPHA)
    stage1.fit(outlet_mrp_features(train_df), units)
    return stage1


def predict_units_stage1(stage1, df):
    """Units predicted by Stage 1, never below MIN_UNITS."""
    units_stage1 = stage1.predict(outlet_mrp_features(df))
    return np.maximum(units_stage1, MIN_UNITS)


def fit_stage2(stage1, train_df):
    """Stage 2: learn a correction factor for each item number.

    ratio = actual units / Stage 1 units
        ratio = 1.10 -> Stage 1 predicted 10% too little for this row
        ratio = 0.90 -> Stage 1 predicted 10% too much for this row
    Stage 2 learns the average ratio for each item number.
    """
    mrp = train_df['Item_MRP'].values
    units = get_units(train_df)
    units_stage1 = predict_units_stage1(stage1, train_df)

    ratio = units / units_stage1

    # RMSE is measured in SALES, so a row with high predicted sales matters more.
    # Weight every row by (predicted sales) squared, then rescale so the average weight is 1.
    weight = (units_stage1 * mrp) ** 2
    weight = weight / weight.mean()

    stage2 = HistGradientBoostingRegressor(**STAGE2_SETTINGS)
    stage2.fit(item_number_feature(train_df), ratio, sample_weight=weight)
    return stage2


def fit_two_stage(train_df):
    """Train both stages on train_df. Returns (stage1, stage2)."""
    stage1 = fit_stage1(train_df)
    stage2 = fit_stage2(stage1, train_df)
    return stage1, stage2


def predict_sales(stage1, stage2, df, use_stage2=True):
    """Predicted sales = Stage 1 units x Stage 2 correction x MRP.
    With use_stage2=False we only use Stage 1 (needed to measure what Stage 2 adds)."""
    units_stage1 = predict_units_stage1(stage1, df)

    if use_stage2:
        correction = stage2.predict(item_number_feature(df))
    else:
        correction = 1.0

    return units_stage1 * correction * df['Item_MRP'].values


# =====================================================================
# STEP 5: CROSS-VALIDATION
# =====================================================================
# Cross-validation = an honest exam for the model:
#   * split the training data into 5 parts (folds)
#   * train on 4 parts, predict the 5th part (which the model has never seen)
#   * repeat until every part has been predicted once
#   * compare the predictions with the real sales
# Both stages are re-trained inside every fold, so nothing from the exam part leaks into training.
print()
print('=' * 60)
print('STEP 5: Cross-validation')
print('=' * 60)


def rmse(actual, predicted):
    """Root mean squared error: square root of the average squared mistake."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def baseline_average(fit_df, val_df):
    """Baseline 1: predict the same average sales for every row."""
    return np.full(len(val_df), fit_df['Item_Outlet_Sales'].mean())


def baseline_one_line(fit_df, val_df):
    """Baseline 2: one straight line sales ~ MRP, the same for all outlets."""
    line = LinearRegression()
    line.fit(fit_df[['Item_MRP']], fit_df['Item_Outlet_Sales'])
    return line.predict(val_df[['Item_MRP']])


def stage1_only(fit_df, val_df):
    """Our Stage 1 alone: one line per outlet."""
    stage1, stage2 = fit_two_stage(fit_df)
    return predict_sales(stage1, stage2, val_df, use_stage2=False)


def final_model(fit_df, val_df):
    """Our final model: Stage 1 + Stage 2."""
    stage1, stage2 = fit_two_stage(fit_df)
    return predict_sales(stage1, stage2, val_df, use_stage2=True)


# name shown in the table -> function that makes the predictions
MODELS = {
    'Baseline 1: predict the average': baseline_average,
    'Baseline 2: one MRP line for all outlets': baseline_one_line,
    'Stage 1 only: one line per outlet': stage1_only,
    'Final: Stage 1 + Stage 2': final_model,
}

actual_sales = train['Item_Outlet_Sales'].values
scores = {name: [] for name in MODELS}            # RMSE of every model for every seed

for seed in CV_SEEDS:
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    # one array per model to collect the predictions for all rows (out-of-fold predictions)
    predictions = {name: np.zeros(len(train)) for name in MODELS}

    for fit_rows, val_rows in splitter.split(train):
        fit_df = train.iloc[fit_rows]             # 4 parts: used for training
        val_df = train.iloc[val_rows]             # 1 part : used for the exam

        for name, model_function in MODELS.items():
            predictions[name][val_rows] = model_function(fit_df, val_df)

    for name in MODELS:
        scores[name].append(rmse(actual_sales, predictions[name]))

print('Cross-validation RMSE (lower is better)')
print('-' * 72)
print(f'{"Model":44s} {"mean":>8s} {"best":>8s} {"worst":>8s}')
print('-' * 72)
for name, seed_scores in scores.items():
    print(f'{name:44s} {np.mean(seed_scores):8.1f} {np.min(seed_scores):8.1f} {np.max(seed_scores):8.1f}')
print('-' * 72)

# Did Stage 2 help on every split? If yes, the gain is not just luck.
gains = [s1 - s2 for s1, s2 in zip(scores['Stage 1 only: one line per outlet'],
                                   scores['Final: Stage 1 + Stage 2'])]
print('Gain of Stage 2 on each split:', [round(g, 1) for g in gains])
print(f'Stage 2 lowers the RMSE by {np.mean(gains):.1f} on average; helped on every split: {all(g > 0 for g in gains)}')


# =====================================================================
# STEP 6: TRAIN ON ALL DATA, PREDICT THE TEST SET, SAVE
# =====================================================================
print()
print('=' * 60)
print('STEP 6: Final training and prediction')
print('=' * 60)

# Now that the models are checked, train them on ALL training rows
stage1, stage2 = fit_two_stage(train)

# Predict the test set
test_predictions = predict_sales(stage1, stage2, test)

# Build the output table: the two id columns and the predicted sales
submission = test_ids.copy()
submission['Item_Outlet_Sales'] = np.maximum(test_predictions, 0)      # sales cannot be negative

submission.to_csv(OUT_PATH, index=False)
print('Saved', OUT_PATH, 'with shape', submission.shape)


# =====================================================================
# STEP 7: CHECK THE SAVED FILE
# =====================================================================
print()
print('=' * 60)
print('STEP 7: Check the saved file')
print('=' * 60)

saved = pd.read_csv(OUT_PATH)

same_order = (saved['Item_Identifier'].values == test['Item_Identifier'].values).all()

print('Number of rows is 5681           :', len(saved) == 5681)
print('No missing values                :', bool(saved['Item_Outlet_Sales'].notnull().all()))
print('No negative sales                :', bool((saved['Item_Outlet_Sales'] >= 0).all()))
print('Same row order as the test file  :', bool(same_order))

print()
print('Predicted sales: min {:.0f} | average {:.0f} | max {:.0f}'.format(
    saved['Item_Outlet_Sales'].min(), saved['Item_Outlet_Sales'].mean(), saved['Item_Outlet_Sales'].max()))
print('Average sales in the training data: {:.0f}'.format(train['Item_Outlet_Sales'].mean()))
