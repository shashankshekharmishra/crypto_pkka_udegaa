import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Prevents GUI issues on headless systems
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Configuration
CRYPTO_SYMBOL = "BTC-USD"  # Change to "ETH-USD", "DOGE-USD", etc.
WINDOW_SIZE = 60
EPOCHS = 20
BATCH_SIZE = 32

# Step 1: Download data
print(f"📥 Downloading data for {CRYPTO_SYMBOL}...")
df = yf.download(CRYPTO_SYMBOL, period="180d")
data = df['Close'].values.reshape(-1, 1)

# Step 2: Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: Prepare sequences
X, y = [], []
for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i-WINDOW_SIZE:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Step 4: Build the model
print("🧠 Building model...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
print("🏋️ Training model...")
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Step 6: Save model
model.save("model_fixed.h5")
print("✅ Model saved as model_fixed.h5")

# Step 7: Plot training loss
print("📊 Saving training loss plot...")
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.title(f'{CRYPTO_SYMBOL} Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png")
print("✅ Training loss plot saved as training_loss.png")
