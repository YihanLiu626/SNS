import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# -------------------------------
# 1. 加载先前训练好的模型和对象
# -------------------------------
# 加载模型（使用 ModelCheckpoint 保存为 best_model_250319.h5）
best_model = tf.keras.models.load_model("best_model_250319.h5")

# 加载 scaler（请确保在训练脚本中将 scaler 保存为 scaler.pkl）
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 加载原始数据（请确保在训练脚本中将数据保存为 data.pkl）
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

# 定义全局变量
time_step = 60
features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50',
            'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Return']

# 使用原始数据和 scaler 重新计算 scaled_data
scaled_data = scaler.transform(data[features].values)


# -------------------------------
# 2. 定义工具函数
# -------------------------------
# 反归一化函数（与训练时使用的相同）
def inverse_transform(predictions, scaler, n_features):
    dummy = np.zeros((predictions.shape[0], n_features - 1))
    pred_full = np.concatenate((predictions, dummy), axis=1)
    return scaler.inverse_transform(pred_full)[:, 0]


# 预测函数：使用最近 time_step 天的数据来预测下一天的收盘价
def predict_next_price():
    global best_model, scaler, scaled_data, time_step, features
    # 获取最近 time_step 天的归一化数据
    last_sequence = scaled_data[-time_step:]
    last_sequence = np.expand_dims(last_sequence, axis=0)  # 形状变为 (1, time_step, num_features)
    prediction_scaled = best_model.predict(last_sequence)
    predicted_price = inverse_transform(prediction_scaled, scaler, len(features))
    return predicted_price[0]


# -------------------------------
# 3. 定义 GUI 界面和回调函数
# -------------------------------
# 回调函数：更新预测结果标签
def show_prediction():
    predicted_price = predict_next_price()
    result_label.config(text=f"Predicted Next Day's Close Price: {predicted_price:.2f}")

# 回调函数：生成图表并在 GUI 中显示
def show_plot():
    global best_model, scaler, scaled_data, time_step, features, data
    predicted_price = predict_next_price()

    # 获取最近 time_step 天的日期和历史收盘价
    historical_dates = data.index[-time_step:]
    historical_close = data['Close'].iloc[-time_step:].values
    # 假设预测日期为最后一个历史日期的后一天
    next_date = historical_dates[-1] + pd.Timedelta(days=1)

    # 分离日期和价格
    hist_dates = list(historical_dates)
    hist_prices = list(historical_close)
    pred_date = next_date
    pred_price = predicted_price

    # 创建 matplotlib 图表和坐标轴
    fig, ax = plt.subplots(figsize=(6, 4))

    # 将历史数据（60 天）绘制为一条曲线
    ax.plot(hist_dates, hist_prices, marker='o', color='blue', label='Historical')

    # 将预测日期作为一个单独的点绘制（例如红色 'X'）
    ax.plot(pred_date, pred_price, marker='x', markersize=10, color='red', label='Prediction')

    # 添加图例、标题和坐标轴标签
    ax.set_title("Historical Close Prices and Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    fig.autofmt_xdate()

    # 将图表嵌入到 Tkinter 窗口中
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    # 清除之前的图表显示（如果有），然后添加新的画布
    for widget in plot_frame.winfo_children():
        widget.destroy()
    canvas.get_tk_widget().pack()


# -------------------------------
# 4. 构建 Tkinter GUI 界面
# -------------------------------
window = tk.Tk()
window.title("Stock Price Prediction")
window.geometry("600x600")

# 标题标签
title_label = ttk.Label(window, text="Stock Price Prediction", font=("Arial", 16))
title_label.pack(pady=10)

# 预测按钮
predict_button = ttk.Button(window, text="Predict Next Day's Close Price", command=show_prediction)
predict_button.pack(pady=10)

# 用于显示预测结果的标签
result_label = ttk.Label(window, text="Prediction result will be displayed here", font=("Arial", 12))
result_label.pack(pady=10)

# 用于显示图表的框架
plot_frame = ttk.Frame(window)
plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)

# 显示图表的按钮
plot_button = ttk.Button(window, text="Show Prediction Chart", command=show_plot)
plot_button.pack(pady=10)

window.mainloop()
