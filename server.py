import io
import base64
import datetime
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 让 matplotlib 以后台模式运行
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, request, jsonify
import pickle
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# 训练时的特征
features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Return']
time_step = 60  # 过去 60 天数据用于预测


def get_data(ticker):
    """ 获取最新的股票数据，并确保至少有 60 天数据 """
    data = yf.download(ticker, period="3mo")  # 获取最近三个月数据，确保有足够的天数


    # 确保数据至少有 60 天
    if len(data) < time_step:
        print(f"⚠️ {ticker} 数据不足 {time_step} 天，仅有 {len(data)} 天数据。使用所有可用数据。")

    return data[-time_step:]  # 总是取最近的 60 天数据


def compute_features(df):
    """ 计算技术指标 """
    df = df.copy()

    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + 2 * df['STD20']
    df['BB_lower'] = df['MA20'] - 2 * df['STD20']

    df['Return'] = df['Close'].pct_change()

    df.dropna(inplace=True)
    return df


def inverse_transform(predictions, scaler, n_features):
    """ 反归一化 """
    dummy = np.zeros((predictions.shape[0], n_features - 1))
    pred_full = np.concatenate((predictions, dummy), axis=1)
    return scaler.inverse_transform(pred_full)[:, 0]


def predict_future_prices(model, recent_data_scaled, scaler, n_days=1):
    """ 使用模型进行未来价格预测 """
    predicted_prices = []
    input_data = recent_data_scaled.copy()

    for _ in range(n_days):
        prediction = model.predict(np.expand_dims(input_data, axis=0))
        prediction_actual = inverse_transform(prediction.reshape(-1, 1), scaler, len(features))
        predicted_prices.append(prediction_actual[0])

        new_entry = np.zeros((1, len(features)))
        new_entry[0, 0] = prediction[0, 0]
        input_data = np.vstack([input_data[1:], new_entry])

    return predicted_prices


@app.route("/predict", methods=["POST"])
def predict():
    """ 处理 Chatbot 请求，返回预测结果和图像 """
    data = request.get_json()
    ticker = data.get("ticker", "AAPL").upper()
    days_to_predict = int(data.get("days", 1))

    if ticker not in ["AAPL", "GOOG","AMZN"]:
        return jsonify({"error": "Only AAPL and TSLA and AMZN are supported"}), 400

    model_file = f"fixed_best_model_{ticker}.h5"
    scaler_file = f"scaler_{ticker}.pkl"

    try:
        model = tf.keras.models.load_model(model_file)
        with open(scaler_file, "rb") as f:
            scaler_new = pickle.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to load model or scaler: {e}"}), 500

    data_new = get_data(ticker)
    data_new = compute_features(data_new)

    scaled_data_new = scaler_new.transform(data_new[features])
    recent_data_scaled = scaled_data_new[-time_step:]

    future_predictions = predict_future_prices(model, recent_data_scaled, scaler_new, days_to_predict)

    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    predicted_dates = pd.date_range(start=tomorrow, periods=days_to_predict, freq='B')

    predictions = {date.strftime("%Y-%m-%d"): round(price, 2)
                   for date, price in zip(predicted_dates, future_predictions)}

    plt.figure(figsize=(6, 4))
    plt.plot(predicted_dates, future_predictions, marker='o', color='red', label='Predicted Prices')
    plt.title(f"{ticker} - Predicted Prices for Next {days_to_predict} Days")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return jsonify({"predictions": predictions, "chart": chart_base64})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
