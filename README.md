# Python Small Project: Weather Data + Linear Regression (From Scratch)

## Overview
This project has two main goals:

1. Collect weather sensor data from Firebase Realtime Database (DHT11).
2. Train and test a simple Linear Regression model written from scratch (without scikit-learn).

The current dataset is stored in `temp_data.csv` and includes:
- Temperature
- Humidity
- Time

## Core Idea
The pipeline is intentionally simple:

1. Read sensor values from Firebase nodes (`Temperature`, `Humidity`, `Time`).
2. Export them into a CSV file.
3. Use the CSV to train a custom Linear Regression model.
4. Evaluate prediction quality with basic metrics.

This project is useful for learning:
- Basic data collection from cloud database
- Data cleaning with pandas
- Gradient Descent implementation
- End-to-end ML testing flow in pure Python

## Project Files
- `FirebaseData.py`: connects to Firebase, reads data nodes, aligns lengths, and exports CSV.
- `LinearRegressionFromScratch.py`: contains the custom Linear Regression class.
- `MainTestModel.py`: test runner that loads CSV, trains model, and prints metrics.
- `temp_data.csv`: exported dataset used for model testing.

## Model Testing Flow
`MainTestModel.py` does the following:

1. Loads `temp_data.csv`.
2. Converts `Temperature` and `Humidity` to numeric values.
3. Drops invalid rows.
4. Splits train/test in time order.
5. Scales input feature for stable training.
6. Trains `LinearRegressionFromScratch`.
7. Prints MAE, RMSE, R2, and sample predictions.

## Notes
- If train output becomes `NaN`, the learning rate may be too high or features are not scaled.
- Current baseline uses only `Humidity` to predict `Temperature`, so performance can be improved by adding features from `Time`.
- Keep your Firebase service account key private.
