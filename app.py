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




def get_tw0050_stocks():
    response = requests.get('https://answerbook.david888.com/TW0050')
    data = response.json()
    
    # 取得股票代碼並加上 .TW
    stocks = [f"{code}.TW" for code in data['stocks'].keys()]
    
    # 如果需要排序的話可以加上 sort()
    #stocks.sort()
    
    return stocks

def get_sp500_full_stocks():

    response = requests.get('https://answerbook.david888.com/SP500')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['stocks'].keys())
    
    return stocks



def get_nasdaq_full_stocks():
    response = requests.get('http://13.125.121.198:8090/stocks/NASDAQ100')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['stocks'].keys())
    
    return stocks



def get_tw0051_stocks():
    response = requests.get('https://answerbook.david888.com/TW0051')
    data = response.json()
    
    # 取得股票代碼並加上 .TW
    stocks = [f"{code}.TW" for code in data['stocks'].keys()]
    
    # 如果需要排序的話可以加上 sort()
    # stocks.sort()
    
    return stocks





def get_sp500_stocks(limit=50):
    response = requests.get('https://answerbook.david888.com/SP500')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['stocks'].keys())[:limit]
    
    return stocks



# Function to fetch NASDAQ component stocks
def get_nasdaq_stocks(limit=50):

    response = requests.get('http://13.125.121.198:8090/stocks/NASDAQ100')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['stocks'].keys())[:limit]
    
    return stocks


# Function to fetch Philadelphia Semiconductor Index component stocks
def get_sox_stocks():
    return [
        "NVDA", "AVGO", "GFS", "CRUS", "ON", "ASML", "QCOM", "SWKS", "MPWR", "ADI",
        "TSM", "AMD", "TXN", "QRVO", "AMKR", "MU", "ARM", "NXPI", "TER", "ENTG",
        "LSCC", "COHR", "ONTO", "MTSI", "KLAC", "LRCX", "MRVL", "AMAT", "INTC", "MCHP"
    ]
# Function to fetch Dow Jones Industrial Average component stocks
def get_dji_stocks():

    response = requests.get('http://13.125.121.198:8090/stocks/DOWJONES')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['stocks'].keys())
    
    return stocks


# Function to get top 10 potential stocks
def get_top_10_potential_stocks(period, selected_indices):
    stock_list = []
    if "台灣50" in selected_indices:
        stock_list += get_tw0050_stocks()
    if "台灣中型100" in selected_indices:
        stock_list += get_tw0051_stocks()
    if "S&P精簡版50" in selected_indices:
        stock_list += get_sp500_stocks()
    if "NASDAQ精簡版50" in selected_indices:
        stock_list += get_nasdaq_stocks()
    if "費城半導體SOX" in selected_indices:
        stock_list += get_sox_stocks()
    if "道瓊DJI" in selected_indices:
        stock_list += get_dji_stocks()
    if "NASDAQ完整版100" in selected_indices:
        stock_list += get_nasdaq_full_stocks()        
    if "S&P完整版500" in selected_indices:
        stock_list += get_sp500_full_stocks()        

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
            potential = (predicted_prices[-1][0] - data['Close'].values[-1]) / data['Close'].values[-1]
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
    gr.CheckboxGroup(choices=["台灣50", "台灣中型100", "S&P精簡版50", "NASDAQ精簡版50", "費城半導體SOX", "道瓊DJI","NASDAQ完整版100", "S&P完整版500"], label="指數選擇", value=["台灣50", "台灣中型100"])
]
outputs = gr.Dataframe(label="潛力股推薦結果")

gr.Interface(fn=stock_prediction_app, inputs=inputs, outputs=outputs, title="台股美股潛力股推薦系統 - LSTM模型")\
    .launch()
