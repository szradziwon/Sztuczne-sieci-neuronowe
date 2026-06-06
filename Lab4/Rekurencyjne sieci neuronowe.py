import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense


def create_sequences(data, sequence_length=50):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)


def build_model(model_type, sequence_length, n_features):
    if model_type == "LSTM":
        layer = LSTM(50, return_sequences=False, input_shape=(sequence_length, n_features))
    elif model_type == "RNN":
        layer = SimpleRNN(50, return_sequences=False, input_shape=(sequence_length, n_features))
    elif model_type == "GRU":
        layer = GRU(50, return_sequences=False, input_shape=(sequence_length, n_features))

    model = Sequential([
        layer,
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


symbol = "AAPL"
data = yf.download(symbol, start="2010-01-01", end="2023-12-31")

data = data[['Close', 'Volume']]
data = data.dropna()
print(data.head())
plt.show()

sequence_lengths = [30, 50, 100]
model_types = ["LSTM", "RNN", "GRU"]
feature_sets = {
    "Close": ['Close'],
    "Close + Volume": ['Close', 'Volume']
}

results = []
best_result = None

for feature_set_name, columns in feature_sets.items():
    prices = data[columns]

    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(data[['Close']])

    for sequence_length in sequence_lengths:
        X, y = create_sequences(prices_scaled, sequence_length)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        for model_type in model_types:
            model = build_model(model_type, sequence_length, len(columns))
            print(f"\nModel: {model_type}, okno: {sequence_length}, dane: {feature_set_name}")
            print(model.summary())

            history = model.fit(
                X_train,
                y_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )

            predicted_prices = model.predict(X_test)

            # Odwrócenie skalowania
            y_test_rescaled = close_scaler.inverse_transform(y_test.reshape(-1, 1))
            predicted_prices_rescaled = close_scaler.inverse_transform(predicted_prices)

            rmse = np.sqrt(mean_squared_error(y_test_rescaled, predicted_prices_rescaled))
            mae = mean_absolute_error(y_test_rescaled, predicted_prices_rescaled)

            results.append({
                "model": model_type,
                "okno": sequence_length,
                "dane": feature_set_name,
                "RMSE": rmse,
                "MAE": mae
            })

            if best_result is None or rmse < best_result["RMSE"]:
                best_result = {
                    "model": model_type,
                    "okno": sequence_length,
                    "dane": feature_set_name,
                    "RMSE": rmse,
                    "MAE": mae,
                    "y_test_rescaled": y_test_rescaled,
                    "predicted_prices_rescaled": predicted_prices_rescaled,
                    "dates": data.index[-len(y_test):]
                }
            if model_type == "LSTM" and sequence_length == 50 and feature_set_name == "Close":
                original_result = {
                    "y_test_rescaled": y_test_rescaled,
                    "predicted_prices_rescaled": predicted_prices_rescaled,
                    "dates": data.index[-len(y_test):],
                    "RMSE": rmse,
                    "MAE": mae
                }

results_df = pd.DataFrame(results).sort_values(by="RMSE")
print("\nPorównanie wyników:")
print(results_df)

# Wizualizacja najlepszego wyniku
plt.figure(figsize=(12, 6))
plt.plot(best_result["dates"], best_result["y_test_rescaled"], label='Rzeczywiste ceny')
plt.plot(original_result["dates"], original_result["predicted_prices_rescaled"], label='Pierwotna predykcja LSTM')
plt.plot(best_result["dates"], best_result["predicted_prices_rescaled"], label='Prognozowane ceny')
plt.legend()
plt.title(
    f'Najlepszy wynik: {best_result["model"]}, '
    f'okno={best_result["okno"]}, dane={best_result["dane"]}, '
    f'RMSE={best_result["RMSE"]:.2f}'
)
plt.xlabel('Data')
plt.ylabel('Cena akcji')
plt.show()
