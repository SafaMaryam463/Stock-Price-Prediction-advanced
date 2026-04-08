from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def advice(pred, current):
    diff = pred - current
    if diff > 5:
        return "BUY 📈"
    elif diff < -5:
        return "SELL 📉"
    else:
        return "HOLD 🤝"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock = request.form['stock']

        df = pd.read_csv(f"stocks/{stock}.csv")

        last_close = df['Close'].iloc[-1]

        scaled = scaler.transform([[last_close]])
        prediction = model.predict(scaled)[0]

        adv = advice(prediction, last_close)

           
        import os

        if not os.path.exists('static'):
            os.makedirs('static')

        plt.style.use('dark_background')

        plt.figure(figsize=(4,2))
        plt.plot(df['Close'], color="#d483e4", linewidth=2)

        plt.fill_between(range(len(df['Close'])), df['Close'], color='#00ffcc', alpha=0.1)

        plt.title("Stock Price Trend")
        plt.xlabel("Days")
        plt.ylabel("Price")

        plt.grid(alpha=0.2)

        plt.savefig(os.path.join('static', 'chart.png'), bbox_inches='tight')
        plt.close()

        return render_template("index.html",
                               stock=stock,
                               last_price=round(last_close, 2),
                               prediction=round(prediction, 2),
                               advice=adv)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)