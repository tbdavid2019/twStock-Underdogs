---
title: TwStock Underdogs
emoji: 🏆
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
pinned: false
short_description: 台灣上市股票 潛力股
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

# Taiwan Stock Short-Term Potential Stock Prediction System - LSTM Model

This project is a stock prediction system for short-term potential stocks in the Taiwan stock market. It uses LSTM (Long Short-Term Memory) models to predict future prices based on historical data from Taiwan listed stocks. The system is implemented using Python, and the web interface is created using Gradio. The project is designed for research and educational purposes only and should not be used for investment decision-making.

## Features
- Fetch historical stock data for Taiwan listed stocks using `yfinance`.
- Train LSTM models on stock data to predict future prices.
- Recommend the top 10 potential stocks based on prediction analysis.
- User-friendly web interface created with Gradio.

## Installation
To run the project locally or on a server like Hugging Face Space, you need to install the required dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/tbdavid2019/twStock-Underdogs.git
   cd twStock-Underdogs
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
You can run the application by executing the `app.py` file.

```bash
python app.py
```
The application will launch a Gradio interface where you can select a period (e.g., 1 month, 3 months) and receive predictions for the top 10 potential stocks.

## Requirements
The `requirements.txt` file includes the necessary libraries for the project:
- `yfinance`
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `gradio`
- `keras`

## Note
This system is for research purposes only. Investing in stocks involves significant risk, and predictions made by this system should not be the sole basis for investment decisions.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

# 台股短線潛力股推薦系統 - LSTM 模型

此專案是一個針對台灣股票市場短線潛力股的預測系統，使用 LSTM（長短期記憶）模型根據台灣上市股票的歷史數據來預測未來股價。本系統使用 Python 實現，並透過 Gradio 建立網頁介面。本專案僅供學術研究用途，不應用於投資決策。

## 功能
- 使用 `yfinance` 抓取台灣上市股票的歷史數據。
- 使用 LSTM 模型進行股價預測。
- 根據預測分析推薦前 10 名潛力股。
- 透過 Gradio 建立簡單易用的網頁介面。

## 安裝
若要在本地或 Hugging Face Space 等伺服器上運行本專案，請先安裝必要的相依套件。

1. 複製此倉庫：
   ```bash
   git clone https://github.com/tbdavid2019/twStock-Underdogs.git
   cd twStock-Underdogs
   ```

2. 安裝相依套件：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方式
執行 `app.py` 來啟動應用程式：

```bash
python app.py
```
應用程式會啟動一個 Gradio 介面，您可以選擇時間範圍（例如：1 個月、3 個月），並查看系統推薦的前 10 名潛力股。

## 相依套件
`requirements.txt` 文件包含了本專案所需的套件：
- `yfinance`
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `gradio`
- `keras`

## 注意事項
此系統僅供研究用途。投資股票具有高度風險，本系統的預測結果不應作為投資決策的唯一依據。

## 授權
本專案使用 MIT 授權條款。詳細請參見 LICENSE 文件。


