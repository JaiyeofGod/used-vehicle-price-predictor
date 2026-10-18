# Used Vehicle Price Predictor

COMP 365 Individual Project

This project estimates the price of a used vehicle. It compares a simple average-price baseline with Linear Regression, a Decision Tree, and a Random Forest. The web app uses the saved Random Forest model.

Live app: https://used-vehicle-price-predictor.streamlit.app/

## Dataset

The dataset is the Used Car Price Prediction Dataset published on Kaggle by Taeef Najib:
https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset

A copy is included as `used_cars.csv` in the project root. To obtain another copy, download the dataset from Kaggle, extract it, and place `used_cars.csv` in the project root.

Put `used_cars.csv` in the project root, in the same folder as `checkpoint1.py` and `app.py`. It is already there. The file has a header and 4,009 rows. The columns are `brand`, `model`, `model_year`, `milage`, `fuel_type`, `engine`, `transmission`, `ext_col`, `int_col`, `accident`, `clean_title`, and `price`.

The mileage column is spelled `milage`. The code depends on that spelling. In the raw file, `price` and `milage` are text and `model_year` is a whole number. The other columns are text. `fuel_type` is missing 170 values, `accident` is missing 113, and `clean_title` is missing 596. The other columns have no missing values. There are 0 duplicate rows.

## Setup

Run these steps from the project root.

`requirements.txt` pins `scikit-learn==1.7.2` and `streamlit==1.65.0`. It also installs `pandas` and `joblib`, but it does not pin their versions. The project does not name a required Python version.

The checks below used Python 3.10.4, pandas 2.3.3, and joblib 1.6.0. That is the environment I checked now. It is separate from the computer used for the original training, which is listed under Hardware.

Create and activate a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

These commands were tested in a separate temporary copy of the project. That fresh environment used Python 3.10.4, pandas 2.3.3, scikit-learn 1.7.2, streamlit 1.65.0, and joblib 1.6.0.

1. Install the dependencies. This command was run in the fresh virtual environment, and the install finished successfully.

```bash
python3 -m pip install -r requirements.txt
```

2. Clean the data, train the models, and repeat the evaluation with one script. There are no separate scripts for those steps. This command was run in the temporary copy, not in this project folder. Every MAE and R² matched the results table. Running it in this folder would still replace `cleaned_used_cars.csv`, `checkpoint2_model_comparison.csv`, and `vehicle_price_model.pkl`.

```bash
python3 checkpoint1.py
```

3. Start the web app. The app reads `used_cars.csv` and `vehicle_price_model.pkl`. It does not train a model. This command was tested in the temporary copy. The server started, a vehicle was submitted, and the app returned an estimated price. To use the app locally, open http://localhost:8501.

```bash
python3 -m streamlit run app.py
```

## Training settings

`checkpoint1.py` reads `used_cars.csv`, removes `$` and commas from `price`, and removes ` mi.` and commas from `milage`. Both become numbers. It counts missing values and duplicates, but it does not delete rows. It saves the cleaned table as `cleaned_used_cars.csv`. That file still has 4,009 rows. Cleaned prices run from 2,000 to 2,954,083 dollars, and cleaned mileage runs from 100 to 405,000.

The target is the cleaned price. Each model group uses an 80/20 split: `test_size=0.2` and `random_state=42`. The script makes that split once for Checkpoint 1 and again for Checkpoint 2. It does not pass any other split settings. The random seed makes the split repeatable.

Checkpoint 1 uses only `model_year` and `milage`. Those two columns have no missing values, so this step does not fill anything in. Linear Regression is `LinearRegression()` with no extra arguments. The non-AI baseline predicts the average training price for every test car.

Checkpoint 2 adds `brand`, `model`, `fuel_type`, `engine`, `transmission`, `ext_col`, `int_col`, `accident`, and `clean_title`. Inside each training pipeline, missing numeric values are filled with the training-set median. Missing text values are filled with the most common training-set value, then one-hot encoded. The preprocessing is fitted only on the training rows. One-hot encoding turns each text choice into its own yes-or-no column. Unknown choices are ignored (`handle_unknown="ignore"`).

Checkpoint 2 Linear Regression is `LinearRegression()` with no extra arguments. The Decision Tree is `DecisionTreeRegressor(random_state=42)`. The Random Forest is `RandomForestRegressor(n_estimators=200, random_state=42)`, so it builds 200 trees. Settings that the script does not change stay at the scikit-learn 1.7.2 defaults. For the tree models, those defaults include `criterion="squared_error"`, `max_depth=None`, `min_samples_split=2`, and `min_samples_leaf=1`. Linear Regression still uses `fit_intercept=True`.

The models are evaluated using mean absolute error (MAE) and R² on the same 802 test rows. The training set contains 3,207 rows. MAE is the average dollar amount a prediction was off. R² shows how much of the price differences the model explains. The script saves the rounded scores in `checkpoint2_model_comparison.csv`. The app uses the Checkpoint 2 Random Forest saved in `vehicle_price_model.pkl`.

## Results

These scores were read from `checkpoint2_model_comparison.csv`. A new run in the temporary copy produced the same scores. The original result file was not replaced.

| Method | MAE (USD) | R² |
| --- | ---: | ---: |
| Non-AI baseline | 35276.16 | -0.0025 |
| Checkpoint 1 Linear Regression | 30869.82 | 0.0288 |
| Checkpoint 2 Linear Regression | 25715.33 | 0.0809 |
| Decision Tree | 21586.37 | 0.0796 |
| Random Forest | 18989.26 | 0.1105 |

MAE is in dollars. The Random Forest has the lowest MAE, so that is the saved model.

## Hardware

The original training computer was a 2020 13-inch MacBook Pro with a 1.4 GHz quad-core Intel Core i5, 8 GB of RAM, and macOS Sequoia 15.8.1.

The training script does not record the computer or the training time, and neither is stored in the saved model. The Python and package versions in Setup are from the environment checked for this README, not from a record made during training.

## Limitations and checks

The best R² is 0.1105, so the selected model still leaves most price differences unexplained. Its MAE is 18,989.26 dollars. Some expensive listings have large prediction errors. The highest recorded price is $2,954,083, but these unusual prices have not been independently verified. The project uses one 80/20 split and does not use cross-validation.

The app can only offer brands, models, years, and other categories that are already in `used_cars.csv`. Mileage is typed in. A category the model has never seen is ignored. `pandas` and `joblib` are not pinned, and no Python version is required in the repository.

Training and evaluation were repeated in a separate copy, and the scores matched the table above. The original model and result files were not changed. A prediction was also submitted in that copy of the app. The public Streamlit app was tested successfully and returned a price prediction.

## AI assistance

ChatGPT helped with some parts of planning, writing, and evaluation figures. Cursor helped with code, UI integration, and bug fixes. Google Stitch helped generate the interface design.
