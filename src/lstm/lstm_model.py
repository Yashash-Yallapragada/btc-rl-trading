# lstm_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    """
    Builds and returns a compiled LSTM model.
    
    Args:
        input_shape (tuple): Shape of the input data (timesteps, features)
        
    Returns:
        model (tf.keras.Model): Compiled LSTM model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)  # Predict next price
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model

if __name__ == "__main__":
    import numpy as np

    # Example usage for debugging / training
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()
# Save model architecture to JSON (optional)
model_json = model.to_json()
with open("models/lstm_model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Cell 6: Model Training with Callbacks
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# Load preprocessed data
X = np.load('data/processed/X.npy')
y = np.load('data/processed/y.npy')

# Train-test split
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Model definition (import from Cell 5 or define here)
model = build_lstm_model(input_shape=X.shape[1:])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
    ModelCheckpoint("models/lstm_best_weights.h5", save_best_only=True, monitor='val_loss', verbose=1)
]

# Training
print("🚀 Starting model training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Save full model
model.save("models/lstm_price_predictor.h5")

# Evaluate performance
y_pred = model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, y_pred)
print(f"📉 Validation MAPE: {mape * 100:.2f}%")
if mape < 0.05:
    print("✅ Target MAPE < 5% achieved!")
else:
    print("⚠️ Target not yet reached. Consider tuning hyperparameters.")
import matplotlib.pyplot as plt

# Plot loss curves
def plot_training_history(history, save_path='plots/lstm_training_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Training & Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(save_path)
    print(f"📊 Training curve saved to {save_path}")
    plt.close()

plot_training_history(history)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from copy import deepcopy

# Load test data
from pathlib import Path
X_test = np.load(Path("data/processed") / "X_test.npy")
y_test = np.load(Path("data/processed") / "y_test.npy")


# Ensure model is loaded/trained
# (Assumes `model` is already defined above in your script)

def compute_permutation_importance(model, X, y, n_repeats=10):
    baseline_preds = model.predict(X).flatten()
    baseline_error = mean_squared_error(y, baseline_preds)

    importances = np.zeros(X.shape[2])  # num_features
    for i in range(X.shape[2]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = deepcopy(X)
            np.random.shuffle(X_permuted[:, :, i])  # shuffle feature i across samples
            permuted_preds = model.predict(X_permuted).flatten()
            permuted_error = mean_squared_error(y, permuted_preds)
            scores.append(permuted_error - baseline_error)
        importances[i] = np.mean(scores)
    return importances

# Compute importances
perm_importances = compute_permutation_importance(model, X_test, y_test)

# Load feature names (assumes they’re stored or can be inferred)
feature_names = [
    'open', 'high', 'low', 'close', 'volume',
    'alma', 'fisher', 'stoch_rsi', 'ema_20', 'rsi',
    'macd', 'adx', 'bollinger_upper', 'bollinger_lower', 'obv'
]

# Plot
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(perm_importances)[::-1]
plt.bar(range(len(perm_importances)), perm_importances[sorted_idx], tick_label=np.array(feature_names)[sorted_idx])
plt.title('Permutation Feature Importance')
plt.ylabel('Increase in MSE after Permutation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(Path("plots") / "permutation_importance.png")
plt.show()
print("saved permutation importance plot")
