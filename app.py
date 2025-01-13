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
# def get_tw0050_stocks():
#     return [
#         "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", "2382.TW", "2303.TW", "2882.TW", "2891.TW", "3711.TW",
#         "2412.TW", "2886.TW", "2884.TW", "1216.TW", "2357.TW", "2885.TW", "2892.TW", "3034.TW", "2890.TW", "2327.TW",
#         "5880.TW", "2345.TW", "3231.TW", "2002.TW", "2880.TW", "3008.TW", "2883.TW", "1303.TW", "4938.TW", "2207.TW",
#         "2887.TW", "2379.TW", "1101.TW", "2603.TW", "2301.TW", "1301.TW", "5871.TW", "3037.TW", "3045.TW", "2912.TW",
#         "3017.TW", "6446.TW", "4904.TW", "3661.TW", "6669.TW", "1326.TW", "5876.TW", "2395.TW", "1590.TW", "6505.TW"
#     ]


def get_tw0050_stocks():
    response = requests.get('https://answerbook.david888.com/TW0050')
    data = response.json()
    
    # 取得股票代碼並加上 .TW
    stocks = [f"{code}.TW" for code in data['stocks'].keys()]
    
    # 如果需要排序的話可以加上 sort()
    #stocks.sort()
    
    return stocks

def get_sp500_full_stocks():
    # return [
    #     "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "BRK.B", "NVDA", "UNH", 
    #     "JNJ", "V", "WMT", "JPM", "MA", "PG", "DIS", "HD", "PYPL", "ADBE",
    #     "NFLX", "CMCSA", "PEP", "KO", "MRK", "INTC", "T", "CRM", "CSCO", "PFE", 
    #     "XOM", "COST", "NKE", "CVX", "WFC", "MCD", "AMGN", "MDT", "IBM", "DHR",
    #     "LLY", "HON", "BA", "MMM", "NEE", "ACN", "UPS", "TMO", "AVGO", "PM",
    #     "CSX", "BKNG", "LIN", "ORCL", "QCOM", "TXN", "RTX", "LOW", "MDLZ", "TMUS",
    #     "ISRG", "GE", "AXP", "CAT", "INTU", "ADP", "SPGI", "AMT", "CL", "REGN",
    #     "GS", "VRTX", "LMT", "NOW", "PLD", "CCI", "AON", "KDP", "CHTR", "MSCI",
    #     "SNPS", "ICE", "APD", "FISV", "ELV", "HCA", "CTAS", "EQIX", "WELL", "TGT",
    #     "BK", "STZ", "GILD", "SCHW", "COP", "SBUX", "ADSK", "ATVI", "ZTS", "MRNA",
    #     "BLK", "PGR", "ORLY", "LRCX", "ECL", "ADI", "IDXX", "ITW", "NOC", "ETN",
    #     "F", "D", "HLT", "PXD", "CARR", "ALB", "WMB", "PSA", "MPC", "TRV",
    #     "KEYS", "ODFL", "FTV", "CLX", "LYB", "HIG", "KMX", "PRU", "AVB", "ARE",
    #     "KIM", "EXR", "LHX", "AMP", "ROK", "VRTX", "CHRW", "SBAC", "WST", "TT",
    #     "HSIC", "FE", "ULTA", "DOV", "CDNS", "FAST", "STLD", "EFX", "CE", "GL",
    #     "TSCO", "MTD", "CBOE", "JBHT", "EIX", "XYL", "VLO", "POOL", "TDY", "BR",
    #     "RSG", "PH", "AEE", "CMS", "XYL", "VTR", "EPAM", "ALGN", "CPT", "HSY",
    #     "STE", "KMI", "AAP", "ES", "TTWO", "LVS", "WRB", "WY", "WDC", "ROL",
    #     "CINF", "MKC", "EMR", "ED", "DPZ", "MSI", "HBAN", "WBA", "MOS", "NEM",
    #     "IEX", "SEE", "GPN", "DLR", "NTAP", "TRMB", "ETR", "NDAQ", "RCL", "AEP",
    #     "HOLX", "LW", "GWW", "NVR", "RF", "PFG", "HPE", "BBY", "MHK", "HPQ",
    #     "OKE", "APA", "ALLE", "L", "BF.B", "TSN", "HST", "TPR", "TPG", "KHC",
    #     "DHI", "IRM", "FMC", "CXO", "AAP", "MKTX", "HII", "MTCH", "CRL", "CPB",
    #     "MRO", "RJF", "JNPR", "NTRS", "LNT", "TXT", "FFIV", "PBCT", "WU", "HAS",
    #     "NWSA", "NWS", "FOXA", "FOX", "K", "NUE", "DISCK", "DISCA", "DISCB", "SYY",
    #     "STT", "AKAM", "FRT", "PKI", "DTE", "PPL", "EVRG", "DVA", "BAX", "PNC",
    #     "BEN", "OMC", "VAR", "AOS", "ZBRA", "ATO", "SJM", "PBCT", "JKHY", "RMD",
    #     "CEG", "GPC", "BLL", "CMG", "CF", "MAS", "GME", "HUM", "HRL", "DG",
    #     "CPRI", "HRB", "EL", "KSS", "SYK", "FL", "CNC", "M", "MGA", "APA"
    # ]
    response = requests.get('https://answerbook.david888.com/SP500')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['stocks'].keys())
    
    return stocks



def get_nasdaq_full_stocks():
    return [
        "AAPL", "NVDA", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "AVGO", "COST", 
        "NFLX", "TMUS", "ASML", "CSCO", "ADBE", "AMD", "PEP", "LIN", "AZN", "ISRG", 
        "INTU", "QCOM", "TXN", "BKNG", "CMCSA", "AMGN", "HON", "ARM", "AMAT", "PDD", 
        "PANW", "ADP", "VRTX", "GILD", "SBUX", "MU", "ADI", "MELI", "MRVL", "LRCX", 
        "CTAS", "CRWD", "INTC", "PYPL", "KLAC", "ABNB", "MDLZ", "CDNS", "REGN", "MAR", 
        "CEG", "SNPS", "FTNT", "DASH", "TEAM", "ORLY", "WDAY", "TTD", "CSX", "ADSK", 
        "CHTR", "PCAR", "ROP", "CPRT", "DDOG", "NXPI", "ROST", "AEP", "MNST", "PAYX", 
        "FANG", "FAST", "KDP", "EA", "ODFL", "LULU", "BKR", "VRSK", "XEL", "CTSH", 
        "EXC", "KHC", "GEHC", "CCEP", "IDXX", "TTWO", "CSGP", "ZS", "MCHP", "DXCM", 
        "ANSS", "ON", "WBD", "MDB", "GFS", "CDW", "BIIB", "ILMN", "MRNA", "DLTR", 
        "WBA"
    ]

# def get_tw0051_stocks():
#     return [
#         "2371.TW", "3533.TW", "2618.TW", "3443.TW", "2347.TW", "3044.TW", "2834.TW", "2385.TW", "1605.TW", "2105.TW",
#         "6239.TW", "6176.TW", "9904.TW", "1519.TW", "9910.TW", "1513.TW", "1229.TW", "9945.TW", "2313.TW", "1477.TW",
#         "3665.TW", "2354.TW", "4958.TW", "8464.TW", "9921.TW", "2812.TW", "2059.TW", "1504.TW", "2542.TW", "6770.TW",
#         "5269.TW", "2344.TW", "3023.TW", "1503.TW", "2049.TW", "2610.TW", "2633.TW", "3036.TW", "2368.TW", "3035.TW",
#         "2027.TW", "9914.TW", "2408.TW", "2809.TW", "1319.TW", "2352.TW", "2337.TW", "2006.TW", "2206.TW", "4763.TW",
#         "3005.TW", "1907.TW", "2915.TW", "1722.TW", "6285.TW", "6472.TW", "6531.TW", "3406.TW", "9958.TW", "9941.TW",
#         "1795.TW", "2201.TW", "9917.TW", "2492.TW", "6890.TW", "2845.TW", "8454.TW", "8046.TW", "6789.TW", "2388.TW",
#         "6526.TW", "1802.TW", "5522.TW", "6592.TW", "2204.TW", "2540.TW", "2539.TW", "3532.TW"
#     ]


def get_tw0051_stocks():
    response = requests.get('https://answerbook.david888.com/TW0051')
    data = response.json()
    
    # 取得股票代碼並加上 .TW
    stocks = [f"{code}.TW" for code in data['stocks'].keys()]
    
    # 如果需要排序的話可以加上 sort()
    # stocks.sort()
    
    return stocks


# Function to fetch S&P 500 component stocks
# def get_sp500_stocks():
#     return [
#         "AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "BRK-B", "JNJ", "V", "WMT",
#         "JPM", "MA", "PG", "NVDA", "UNH", "DIS", "HD", "PYPL", "VZ", "ADBE",
#         "NFLX", "CMCSA", "PEP", "KO", "MRK", "INTC", "T", "CRM", "CSCO", "PFE",
#         "XOM", "COST", "NKE", "CVX", "WFC", "MCD", "AMGN", "MDT", "IBM", "DHR",
#         "LLY", "HON", "BA", "MMM", "NEE", "ACN", "UPS", "TMO", "AVGO", "PM"
#     ]


def get_sp500_stocks(limit=50):
    response = requests.get('https://answerbook.david888.com/SP500')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['stocks'].keys())[:limit]
    
    return stocks



# Function to fetch NASDAQ component stocks
def get_nasdaq_stocks():
    # return [
    #     "AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "GOOG", "FB", "NVDA", "PYPL", "ADBE",
    #     "CMCSA", "NFLX", "COST", "PEP", "CSCO", "INTC", "TXN", "AVGO", "AMGN", "QCOM",
    #     "CHTR", "TMUS", "SBUX", "MDLZ", "ISRG", "BKNG", "MRNA", "FISV", "CSX", "ADI",
    #     "VRTX", "ATVI", "GILD", "ILMN", "ADP", "MU", "KLAC", "LRCX", "EA", "KHC",
    #     "JD", "MAR", "BIDU", "MELI", "ROST", "NXPI", "SPLK", "ALGN", "DOCU", "PDD"
    # ]
    response = requests.get('http://13.125.121.198:8090/stocks/NASDAQ100')
    data = response.json()
    
    # 取得股票代碼列表並限制數量
    stocks = list(data['stocks'].keys())
    
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
    # return [
    #     "AAPL", "NVDA", "MSFT", "AMZN", "WMT", "JPM", "V", "UNH", "HD", "PG",
    #     "JNJ", "CRM", "CVX", "KO", "MRK", "CSCO", "IBM", "MCD", "AXP", "DIS",
    #     "GS", "CAT", "VZ", "AMGN", "HON", "BA", "NKE", "SHW", "MMM", "TRV"
    # ]
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
    if "NASDAQ精簡版30" in selected_indices:
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
    gr.CheckboxGroup(choices=["台灣50", "台灣中型100", "S&P精簡版30", "NASDAQ精簡版30", "費城半導體SOX", "道瓊DJI","NASDAQ完整版100", "S&P完整版500"], label="指數選擇", value=["台灣50", "台灣中型100"])
]
outputs = gr.Dataframe(label="潛力股推薦結果")

gr.Interface(fn=stock_prediction_app, inputs=inputs, outputs=outputs, title="台股美股潛力股推薦系統 - LSTM模型")\
    .launch()
