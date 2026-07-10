import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from config import PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
X_test = np.load(PROCESSED_DATA_DIR / 'X_test.npy')
y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')

# Load model without compilation
model = load_model(MODELS_DIR / 'lstm_best_weights.h5', compile=False)
model.compile(optimizer='adam', loss='mse')

# Predict
y_pred = model.predict(X_test)

# Flatten if needed
y_test = y_test.reshape(-1)
y_pred = y_pred.reshape(-1)

# --- Evaluation Metrics ---
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

# Directional Accuracy
direction_true = np.sign(np.diff(y_test))
direction_pred = np.sign(np.diff(y_pred))
directional_accuracy = np.mean(direction_true == direction_pred) * 100

# Print metrics
print("📊 LSTM Model Evaluation Metrics")
print(f"MSE   : {mse:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"MAPE  : {mape:.2f}%")
print(f"R²    : {r2:.4f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# --- Plot: Prediction vs Actual ---
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price', color='orange')
plt.title('Actual vs Predicted BTC Price (Normalized)')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'lstm_prediction_vs_actual.png')
plt.show()

# --- Plot: Residuals ---
residuals = y_test - y_pred
plt.figure(figsize=(12, 4))
plt.plot(residuals, color='red')
plt.title('Residuals (y_test - y_pred)')
plt.xlabel('Time Steps')
plt.ylabel('Error')
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'lstm_residuals.png')
plt.show()

# --- Feature Importance Note ---
print("\n⚠️ Feature importance is not directly available for LSTM models.\n"
      "For explainability, consider SHAP, LIME, or attention mechanisms.\n")
