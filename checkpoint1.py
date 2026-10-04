import pandas as pd

df = pd.read_csv("used_cars.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Clean price column
df["price"] = (
    df["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Clean mileage column
df["milage"] = (
    df["milage"]
    .str.replace(" mi.", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

print("\nCleaned Price and Mileage:")
print(df[["price", "milage"]].head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Select features and target
X = df[["model_year", "milage"]]
y = df["price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nLinear Regression Results:")
print("MAE:", round(mae, 2))
print("R²:", round(r2, 4))

results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": predictions
})

print("\nSample Predictions:")
print(results.head(10).round(2))

# Non-AI baseline: predict the average training-set price
baseline_price = y_train.mean()
baseline_predictions = [baseline_price] * len(y_test)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_r2 = r2_score(y_test, baseline_predictions)

print("\nNon-AI Baseline Results:")
print("Average Price:", round(baseline_price, 2))
print("MAE:", round(baseline_mae, 2))
print("R²:", round(baseline_r2, 4))

df.to_csv("cleaned_used_cars.csv", index=False)

print("\nCleaned dataset saved as cleaned_used_cars.csv")

print("\nFeatures available for Checkpoint 2:")
print(df.columns.tolist())


# Checkpoint 2 starts here: test more vehicle features and compare models

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Select the features for the models
numeric_features = [
    "model_year",
    "milage"
]

categorical_features = [
    "brand",
    "model",
    "fuel_type",
    "engine",
    "transmission",
    "ext_col",
    "int_col",
    "accident",
    "clean_title"
]

features = numeric_features + categorical_features

X2 = df[features].copy()
y2 = df["price"]

print("\nCheckpoint 2 Features:")
print(features)

# Handle missing values and text data
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

# Split the data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2,
    y2,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression with the additional features
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X2_train, y2_train)

linear_predictions = linear_model.predict(X2_test)

linear_mae = mean_absolute_error(
    y2_test,
    linear_predictions
)

linear_r2 = r2_score(
    y2_test,
    linear_predictions
)

print("\nCheckpoint 2 Linear Regression Results:")
print("MAE:", round(linear_mae, 2))
print("R²:", round(linear_r2, 4))

# Train the Decision Tree model
decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=42))
    ]
)

decision_tree_model.fit(
    X2_train,
    y2_train
)

decision_tree_predictions = decision_tree_model.predict(
    X2_test
)

decision_tree_mae = mean_absolute_error(
    y2_test,
    decision_tree_predictions
)

decision_tree_r2 = r2_score(
    y2_test,
    decision_tree_predictions
)

print("\nDecision Tree Results:")
print("MAE:", round(decision_tree_mae, 2))
print("R²:", round(decision_tree_r2, 4))

# Train the Random Forest model
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
            n_estimators=200,
            random_state=42
            )
        )
    ]
)

random_forest_model.fit(
    X2_train,
    y2_train
)

random_forest_predictions = random_forest_model.predict(
    X2_test
)

random_forest_mae = mean_absolute_error(
    y2_test,
    random_forest_predictions
)

random_forest_r2 = r2_score(
    y2_test,
    random_forest_predictions
)

print("\nRandom Forest Results:")
print("MAE:", round(random_forest_mae, 2))
print("R²:", round(random_forest_r2, 4))

# Compare the results from each model
comparison = pd.DataFrame({
    "Method": [
        "Non-AI Baseline",
        "Checkpoint 1 Linear Regression",
        "Checkpoint 2 Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [
        baseline_mae,
        mae,
        linear_mae,
        decision_tree_mae,
        random_forest_mae
    ],
    "R²": [
        baseline_r2,
        r2,
        linear_r2,
        decision_tree_r2,
        random_forest_r2
    ]
})

comparison["MAE"] = comparison["MAE"].round(2)
comparison["R²"] = comparison["R²"].round(4)

print("\nCheckpoint 2 Model Comparison:")
print(comparison.to_string(index=False))

# Show some predictions from the Random Forest model
sample_results = pd.DataFrame({
    "Actual Price": y2_test,
    "Predicted Price": random_forest_predictions
})

print("\nRandom Forest Sample Predictions:")
print(sample_results.head(10).round(2))

# Save the model results
comparison.to_csv(
    "checkpoint2_model_comparison.csv",
    index=False
)

# Check how many unique values the remaining features have
print("\nUnique values in remaining features:")
print("Model:", df["model"].nunique())
print("Engine:", df["engine"].nunique())
print("Exterior color:", df["ext_col"].nunique())
print("Interior color:", df["int_col"].nunique())

import joblib

# Save the Random Forest model for the web app
joblib.dump(random_forest_model, "vehicle_price_model.pkl")

print("\nRandom Forest model saved as vehicle_price_model.pkl")