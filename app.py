from flask import Flask, render_template, request
from model import predict_prices

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prices = []
    plot_url = None
    selected_coin = "BTC-USD"

    if request.method == 'POST':
        selected_coin = request.form['crypto']
        days = int(request.form['days'])
        prices, plot_url = predict_prices(selected_coin, days)

    return render_template('index.html', prices=prices, plot_url=plot_url, crypto=selected_coin)

if __name__ == '__main__':
    # 🟢 Make sure the server starts
    app.run(host='0.0.0.0', port=5000, debug=True)
