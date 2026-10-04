# Used Vehicle Price Prediction System

COMP 365 Individual Project

## Live Web App

https://used-vehicle-price-predictor.streamlit.app/

The deployed application allows users to enter vehicle information and receive a predicted used-vehicle price from the trained Random Forest model.

This project uses machine learning to estimate used-vehicle prices based on vehicle information such as brand, model, year, mileage, engine, transmission, and vehicle history.

## Checkpoint 1

- Loaded a dataset containing 4,009 used vehicles
- Checked missing values and duplicate records
- Cleaned the price and mileage columns
- Created an 80/20 training and testing split
- Trained the first Linear Regression model
- Created a non-AI average-price baseline
- Evaluated the results using MAE and R²

### Checkpoint 1 Results

Linear Regression MAE: $30,869.82  
Linear Regression R²: 0.0288

Non-AI Baseline MAE: $35,276.16  
Non-AI Baseline R²: -0.0025

## Checkpoint 2

- Added more vehicle features to the prediction models
- Handled missing values and categorical data
- Tested Linear Regression, Decision Tree, and Random Forest
- Compared each model using MAE and R²
- Selected Random Forest as the current best model based on MAE
- Built a Streamlit web application
- Connected the trained model to the web application
- Added dropdown filtering so vehicle options match the selected brand and model

### Checkpoint 2 Results

Linear Regression MAE: $25,715.33  
Linear Regression R²: 0.0809

Decision Tree MAE: $21,586.37  
Decision Tree R²: 0.0796

Random Forest MAE: $18,989.26  
Random Forest R²: 0.1105

The Random Forest model currently has the lowest MAE and is being used by the web application.

## Run the Web App

Install the required libraries:

```bash
python3 -m pip install -r requirements.txt