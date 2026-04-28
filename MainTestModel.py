import numpy as np
import pandas as pd
import argparse

from LinearRegressionFromScratch import LinearRegressionFromScratch


def train_test_split_time_order(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
	split_idx = int(len(X) * (1 - test_ratio))
	return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
	mae = np.mean(np.abs(y_true - y_pred))
	rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

	y_mean = np.mean(y_true)
	ss_tot = np.sum((y_true - y_mean) ** 2)
	ss_res = np.sum((y_true - y_pred) ** 2)
	r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

	return mae, rmse, r2


def parse_args():
	parser = argparse.ArgumentParser(description="Test LinearRegressionFromScratch with temp_data.csv")
	parser.add_argument("--humidity", type=float, default=None, help="Single humidity value for prediction")
	parser.add_argument("--actual-temp", type=float, default=None, help="Optional actual temperature to compare")
	return parser.parse_args()


def main():
	args = parse_args()

	df = pd.read_csv("temp_data.csv")

	required_cols = {"Temperature", "Humidity"}
	if not required_cols.issubset(df.columns):
		raise ValueError(f"CSV must contain columns: {required_cols}")

	# Keep only rows where numeric conversion succeeds
	df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
	df["Humidity"] = pd.to_numeric(df["Humidity"], errors="coerce")
	df = df.dropna(subset=["Temperature", "Humidity"]).reset_index(drop=True)

	X = df[["Humidity"]].values
	y = df["Temperature"].values

	X_train, X_test, y_train, y_test = train_test_split_time_order(X, y, test_ratio=0.2)

	# Standardize feature for stable gradient descent
	x_mean = X_train.mean(axis=0)
	x_std = X_train.std(axis=0)
	x_std[x_std == 0] = 1.0

	X_train_scaled = (X_train - x_mean) / x_std
	X_test_scaled = (X_test - x_mean) / x_std

	model = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=3000)
	model.fit(X_train_scaled, y_train)
	y_pred = model.predict(X_test_scaled)

	mae, rmse, r2 = metrics(y_test, y_pred)

	print("=== LinearRegressionFromScratch Test ===")
	print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
	print(f"Weight: {model.weights}")
	print(f"Bias: {model.bias}")
	print(f"MAE: {mae:.4f}")
	print(f"RMSE: {rmse:.4f}")
	print(f"R2: {r2:.4f}")

	if args.humidity is not None:
		single_x = np.array([[args.humidity]])
		single_x_scaled = (single_x - x_mean) / x_std
		single_pred = model.predict(single_x_scaled)[0]

		print("\nSingle input prediction:")
		print(f"Humidity input: {args.humidity}")
		print(f"Predicted Temperature: {single_pred:.4f}")
		if args.actual_temp is not None:
			error = abs(args.actual_temp - single_pred)
			print(f"Actual Temperature: {args.actual_temp}")
			print(f"Absolute Error: {error:.4f}")
	else:
		preview = pd.DataFrame({
			"Humidity": X_test.flatten(),
			"ActualTemp": y_test,
			"PredTemp": y_pred,
		})
		print("\nSample predictions:")
		print(preview.head(10).to_string(index=False))


if __name__ == "__main__":
	main()
