import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import gradio as gr
import datetime

# Function to fetch stock data
def get_stock_data(ticker, period):
    data = yf.download(ticker, period=period)
    return data

# Function to prepare the data for LSTM
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Function to build and train LSTM model
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    return model

# Function to predict stock prices
def predict_stock(model, data, scaler, time_step=60):
    inputs = scaler.transform(data['Close'].values.reshape(-1, 1))
    X_test = []
    for i in range(time_step, len(inputs)):
        X_test.append(inputs[i-time_step:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    return predicted_prices

# Function to fetch Taiwan 50 and Small 100 stocks
def get_tw0050_stocks():
    return [
        "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", "2382.TW", "2303.TW", "2882.TW", "2891.TW", "3711.TW",
        "2412.TW", "2886.TW", "2884.TW", "1216.TW", "2357.TW", "2885.TW", "2892.TW", "3034.TW", "2890.TW", "2327.TW",
        "5880.TW", "2345.TW", "3231.TW", "2002.TW", "2880.TW", "3008.TW", "2883.TW", "1303.TW", "4938.TW", "2207.TW",
        "2887.TW", "2379.TW", "1101.TW", "2603.TW", "2301.TW", "1301.TW", "5871.TW", "3037.TW", "3045.TW", "2912.TW",
        "3017.TW", "6446.TW", "4904.TW", "3661.TW", "6669.TW", "1326.TW", "5876.TW", "2395.TW", "1590.TW", "6505.TW"
    ]

def get_tw0051_stocks():
    return [
        "2371.TW", "3533.TW", "2618.TW", "3443.TW", "2347.TW", "3044.TW", "2834.TW", "2385.TW", "1605.TW", "2105.TW",
        "6239.TW", "6176.TW", "9904.TW", "1519.TW", "9910.TW", "1513.TW", "1229.TW", "9945.TW", "2313.TW", "1477.TW",
        "3665.TW", "2354.TW", "4958.TW", "8464.TW", "9921.TW", "2812.TW", "2059.TW", "1504.TW", "2542.TW", "6770.TW",
        "5269.TW", "2344.TW", "3023.TW", "1503.TW", "2049.TW", "2610.TW", "2633.TW", "3036.TW", "2368.TW", "3035.TW",
        "2027.TW", "9914.TW", "2408.TW", "2809.TW", "1319.TW", "2352.TW", "2337.TW", "2006.TW", "2206.TW", "4763.TW",
        "3005.TW", "1907.TW", "2915.TW", "1722.TW", "6285.TW", "6472.TW", "6531.TW", "3406.TW", "9958.TW", "9941.TW",
        "1795.TW", "2201.TW", "9917.TW", "2492.TW", "6890.TW", "2845.TW", "8454.TW", "8046.TW", "6789.TW", "2388.TW",
        "6526.TW", "1802.TW", "5522.TW", "6592.TW", "2204.TW", "2540.TW", "2539.TW", "3532.TW"
    ]

# Function to get top 10 potential stocks
def get_top_10_potential_stocks(period, selected_indices):
    stock_list = []
    if "tw0050台灣50" in selected_indices:
        stock_list += get_tw0050_stocks()
    if "tw0051中型100" in selected_indices:
        stock_list += get_tw0051_stocks()
    if "S&P" in selected_indices:
        stock_list.append("^GSPC")
    if "NASDAQ" in selected_indices:
        stock_list.append("^IXIC")
    if "費城半導體" in selected_indices:
        stock_list.append("^SOX")
    if "道瓊" in selected_indices:
        stock_list.append("^DJI")

    stock_predictions = []
    time_step = 60

    for ticker in stock_list:
        data = get_stock_data(ticker, period)
        if data.empty or len(data) < time_step:
            # 如果數據為空或不足以生成訓練樣本，則跳過該股票
            continue

        try:
            # Prepare data
            X_train, y_train, scaler = prepare_data(data, time_step=time_step)

            # Train model
            model = train_lstm_model(X_train, y_train)

            # Predict future prices
            predicted_prices = predict_stock(model, data, scaler, time_step=time_step)

            # Calculate the potential (e.g., last predicted price vs last actual price)
            potential = (predicted_prices[-1] - data['Close'].values[-1]) / data['Close'].values[-1]
            stock_predictions.append((ticker, potential, data['Close'].values[-1], predicted_prices[-1][0]))
        except Exception as e:
            print(f"股票 {ticker} 發生錯誤: {str(e)}")
            continue

    # Sort by potential and get top 10
    top_10_stocks = sorted(stock_predictions, key=lambda x: x[1], reverse=True)[:10]
    return top_10_stocks

# Gradio interface function
def stock_prediction_app(period, selected_indices):
    # Get top 10 potential stocks
    top_10_stocks = get_top_10_potential_stocks(period, selected_indices)
    
    # Create a dataframe for display
    df = pd.DataFrame(top_10_stocks, columns=["股票代號", "潛力 (百分比)", "現價", "預測價格"])
    
    return df

# Define Gradio interface
inputs = [
    gr.Dropdown(choices=["3mo", "6mo", "9mo", "1yr"], label="時間範圍"),
    gr.CheckboxGroup(choices=["tw0050台灣50", "tw0051中型100", "S&P", "NASDAQ", "費城半導體", "道瓊"], label="指數選擇")
]
outputs = gr.Dataframe(label="潛力股推薦結果")

gr.Interface(fn=stock_prediction_app, inputs=inputs, outputs=outputs, title="潛力股推薦系統 - LSTM模型")\
    .launch()
