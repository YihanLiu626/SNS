import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt

# 1. 获取数据
def get_data(ticker, start_date="2010-01-01", end_date="2025-01-31"):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# 用户输入股票代码
ticker = input("请输入股票代码（AAPL、GOOG、AMZN、TSLA）：").strip().upper()
data = get_data(ticker)

# 2. 特征工程：计算多种技术指标
def compute_features(df):
    df = df.copy()
    # 移动平均线
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # RSI（14日）
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    # rs = avg_gain / avg_loss
    rs = avg_gain / (avg_loss + 1e-10)  # 避免除以零
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD（使用12日与26日指数移动平均线）
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    # Bollinger Bands（20日）
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + 2 * df['STD20']
    df['BB_lower'] = df['MA20'] - 2 * df['STD20']

    # 日收益率
    df['Return'] = df['Close'].pct_change()

    # 删除因计算指标产生的NaN行
    df.dropna(inplace=True)

    return df

data = compute_features(data)

# 3. 数据预处理与归一化
# 选取特征：收盘价、开盘价、最高价、最低价、成交量、MA_10、MA_50、RSI、MACD、BB_upper、BB_lower、Return
features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Return']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])


# 4. 创建时间序列数据集
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, :])  # 利用过去 time_step 天的所有特征
        y.append(data[i, 0])  # 预测当天的收盘价
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_data, time_step)

# 5. 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 6. 构建模型，利用 keras-tuner 进行超参数调优
def build_model(hp):
    model = Sequential()
    # LSTM 层：搜索 LSTM 单元数
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=dropout_rate))

    # GRU 层：搜索 GRU 单元数
    gru_units = hp.Int('gru_units', min_value=32, max_value=128, step=32)
    model.add(GRU(units=gru_units, return_sequences=False))
    dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=dropout_rate_2))

    # BatchNormalization 层
    model.add(BatchNormalization())

    # 全连接层
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='kt_dir',
    project_name='stock_prediction'
)

tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test),
             callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# 7. 使用最佳模型进行训练
history = best_model.fit(X_train, y_train, epochs=30, batch_size=32,
                         validation_data=(X_test, y_test),
                         callbacks=[
                             EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                             ModelCheckpoint('best_model_250319.h5', monitor='val_loss', save_best_only=True)
                         ])

# 绘制训练过程
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 8. 预测并反归一化
predictions = best_model.predict(X_test).reshape(-1, 1)

# 构造与原始特征数一致的数组用于反归一化（其余特征填充0）
def inverse_transform(predictions, scaler, n_features):
    dummy = np.zeros((predictions.shape[0], n_features - 1))
    pred_full = np.concatenate((predictions, dummy), axis=1)
    return scaler.inverse_transform(pred_full)[:, 0]

predictions_actual = inverse_transform(predictions, scaler, len(features))
y_test_actual = inverse_transform(y_test.reshape(-1, 1), scaler, len(features))

# 9. 可视化预测结果
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
plt.plot(predictions_actual, color='red', label='Predicted Stock Price')
plt.title('Optimized Stock Price Prediction with Enhanced Features & Hyperparameter Tuning')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# 10. 评估模型 - 增加更多评估指标
mse = mean_squared_error(y_test_actual, predictions_actual)
mae = mean_absolute_error(y_test_actual, predictions_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, predictions_actual)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Coefficient of Determination (R²): {r2}')

# 保存 scaler 和原始数据到本地，供 GUI 使用
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)


# 加载已训练的最佳模型和 scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

best_model = tf.keras.models.load_model("best_model_250319.h5")

# 只选择用于训练的相同特征集
features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Return']


def inverse_transform(predictions, scaler, n_features):
    dummy = np.zeros((predictions.shape[0], n_features - 1))
    pred_full = np.concatenate((predictions, dummy), axis=1)
    return scaler.inverse_transform(pred_full)[:, 0]


def predict_future_prices(model, recent_data, scaler, n_days=1):
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


# 让用户选择预测天数
while True:
    try:
        days_to_predict = int(input("请输入要预测的天数（1、5、10、30）："))
        if days_to_predict not in [1, 5, 10, 30]:
            raise ValueError("请输入有效的天数: 1, 5, 10, 30.")
        break
    except ValueError as e:
        print(e)

# 获取最近的时间窗口数据进行预测
recent_data = data[features].iloc[-60:, :].values
recent_data_scaled = scaler.transform(recent_data)

# 进行预测
future_predictions = predict_future_prices(best_model, recent_data_scaled, scaler, days_to_predict)

# 生成日期索引
last_date = data.index[-1]
predicted_dates = pd.date_range(start=last_date, periods=days_to_predict + 1, freq='B')[1:]

# 绘制预测趋势图
plt.figure(figsize=(12, 6))
plt.plot(predicted_dates, future_predictions, marker='o', linestyle='-', color='red', label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'Predicted Stock Prices for the Next {days_to_predict} Days')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 输出预测结果
print(f"未来 {days_to_predict} 天的预测股价:")
for date, price in zip(predicted_dates, future_predictions):
    print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")