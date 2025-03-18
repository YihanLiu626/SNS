import sys
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidgetItem, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from gui import Ui_MainWindow  # 确保 UI 代码已经正确转换为 gui.py

# 设置基础路径，确保所有文件都能被正确加载
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 加载数据处理和模型
def load_resources():
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "data.pkl"), "rb") as f:
        data = pickle.load(f)

    # 加载修正后的 LSTM 模型
    model_path = os.path.join(BASE_DIR, "fixed_best_model.h5")
    if not os.path.exists(model_path):
        QMessageBox.critical(None, "Model Error", f"Model file {model_path} not found!")
        sys.exit(1)

    best_model = tf.keras.models.load_model(model_path)
    return scaler, data, best_model


# 反归一化
def inverse_transform(predictions, scaler, n_features):
    dummy = np.zeros((predictions.shape[0], n_features - 1))
    pred_full = np.concatenate((predictions, dummy), axis=1)
    return scaler.inverse_transform(pred_full)[:, 0]


# 进行未来股价预测（✅ 现在接收 `features` 参数）
def predict_future_prices(model, recent_data, scaler, features, n_days=1):
    predicted_prices = []
    input_data = recent_data.copy()

    for _ in range(n_days):
        prediction = model.predict(np.expand_dims(input_data, axis=0))
        prediction_actual = inverse_transform(prediction.reshape(-1, 1), scaler, len(features))
        predicted_prices.append(prediction_actual[0])

        # 更新输入数据，将新预测值加入并移除最旧的数据点
        new_entry = np.zeros((1, len(features)))
        new_entry[0, 0] = prediction[0, 0]  # 只更新收盘价
        input_data = np.vstack([input_data[1:], new_entry])

    return predicted_prices


# GUI 主窗口类
class StockPredictor(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 绑定预测按钮
        self.btn_predict.clicked.connect(self.run_prediction)

        # 加载模型和数据
        self.scaler, self.data, self.model = load_resources()

        # 设定默认的特征列（✅ 现在 `features` 作为类变量）
        self.features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'RSI', 'MACD', 'BB_upper',
                         'BB_lower', 'Return']

        # **🎯 添加 Matplotlib 画布到 `plot_widget`**
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self.plot_widget)
        layout.addWidget(self.canvas)

    def run_prediction(self):
        # 获取用户输入的股票代码和预测天数
        stock_ticker = self.lineEdit_stock.text().strip().upper()
        days_to_predict = int(self.comboBox_days.currentText())

        if not stock_ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a valid stock ticker.")
            return

        # 获取最近60天数据
        recent_data = self.data[self.features].iloc[-60:, :].values
        recent_data_scaled = self.scaler.transform(recent_data)

        # 进行预测（✅ 传入 `self.features` 以避免 `NameError`）
        future_predictions = predict_future_prices(self.model, recent_data_scaled, self.scaler, self.features,
                                                   days_to_predict)


        # 获取最新市场交易日
        last_date = self.data.index[-1]  # ✅ 使用 self.data

        # 确保 last_date 是最新的实际市场交易日
        from pandas.tseries.offsets import BDay

        today = pd.Timestamp.today()  # 获取今天的日期
        if last_date < today:  # 只更新到今天
            last_date = today

        # 生成未来的交易日
        predicted_dates = pd.date_range(start=last_date, periods=days_to_predict + 1, freq=BDay())[1:]


        # 更新 GUI 结果
        self.label_result.setText(f"Predicted Price: ${future_predictions[-1]:.2f}")

        # 填充表格
        self.table_predictions.setRowCount(len(predicted_dates))
        for i, (date, price) in enumerate(zip(predicted_dates, future_predictions)):
            self.table_predictions.setItem(i, 0, QTableWidgetItem(str(date.date())))
            self.table_predictions.setItem(i, 1, QTableWidgetItem(f"${price:.2f}"))



        # **🎯 更新 Matplotlib 预测曲线**
        self.ax.clear()
        self.ax.plot(predicted_dates, future_predictions, marker='o', linestyle='-', color='red',
                     label='Predicted Prices')

        # **设置 X 轴标签旋转，防止重叠**
        self.ax.set_xticks(predicted_dates)
        self.ax.set_xticklabels(predicted_dates.strftime('%Y-%m-%d'), rotation=45, ha='right')

        # 设置标题、轴标签
        self.ax.set_title(f'Predicted Stock Prices for {days_to_predict} Days')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Stock Price')
        self.ax.legend()
        self.ax.grid()
        import matplotlib.dates as mdates

        # **只显示每 5 天一个日期**
        self.ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


        self.canvas.draw()
        # **强制 Matplotlib 重新绘制**
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictor()
    window.show()
    sys.exit(app.exec())