import yfinance as yf
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64

def predict_prices(symbol, days):
    model = load_model("model_fixed.h5")
    df = yf.download(symbol, period='60d')
    data = df['Close'].values

    scaler = MinMaxScaler()
    scaler_max = float(np.max(data))
    scaler.fit([[0], [scaler_max]])

    scaled_data = scaler.transform(data.reshape(-1, 1))
    window_size = 60
    last_window = scaled_data[-window_size:]

    predictions = []
    input_seq = last_window.reshape(1, window_size, 1)

    for _ in range(days):
        next_price = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(next_price)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_price]]], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()

    # Plot graph
    plt.figure()
    plt.plot(range(1, days + 1), predicted_prices, marker='o')
    plt.title(f'{symbol} Predicted Prices')
    plt.xlabel('Day')
    plt.ylabel('Predicted Close Price (USD)')
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return predicted_prices, plot_data
