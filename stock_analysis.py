"""
title: Stock analysis for stocks listed on BSE and NSE
author: Bhadresh Patel
version: 0.1
requirements: yfinance, pandas, numpy, textblob, fake_useragent, tika, bs4
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import os
import requests
import subprocess
import yfinance as yf
import json
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from textblob import TextBlob
from tika import parser
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

# Initialize Tika
from tika import initVM

initVM()

# Random user agent generator
ua = UserAgent()


def fetch_html(ticker):
    stock_name = ticker.replace(".NS", "").replace(".BO", "")
    url = f"https://www.screener.in/company/{stock_name}/consolidated"
    headers = {"User-Agent": ua.random}
    response = requests.get(url, headers=headers)
    return response.text if response.status_code == 200 else None


def extract_quarterly_result(html_data, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_CONTEXT):
    soup = BeautifulSoup(html_data, "html.parser")
    # Look for the section with id 'shareholding' and class 'card card-large'
    quarterly_result_section = soup.find(
        "section", {"id": "quarters", "class": "card card-large"}
    )

    LLM_summary_request = (
        f"You are a stock market expert. Prepare a detailed json of quarterly results group by the date as output for example quarterly_result->quarter date->sales,expenses,etc. based on below HTML which contains this data in an html table format "
        f"Do not include any tags like ```json or explanation as this output will be directly consumed in a Python program\n\n{quarterly_result_section}"
    )

    OPTIONS = {"num_ctx": OLLAMA_CONTEXT}
    STREAM = False

    response = requests.post(
        url=f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": f"{OLLAMA_MODEL}",
            "messages": [{"role": "user", "content": LLM_summary_request}],
            "options": OPTIONS,
            "stream": STREAM,
        },
    )

    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "No summary received.")
    else:
        print(f"LLM request failed with status code {response.status_code}")
        return "Failed to get summary from LLM."


def extract_profit_loss_result(
    html_data, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_CONTEXT
):
    soup = BeautifulSoup(html_data, "html.parser")
    # Look for the section with id 'shareholding' and class 'card card-large'
    profit_loss_section = soup.find(
        "section", {"id": "profit-loss", "class": "card card-large"}
    )

    LLM_summary_request = (
        f"You are a stock market expert. Prepare a detailed json of profit and loss group by the date as output for example profit_loss_results->date->sales,expenses,etc. based on below HTML which contains this data in an html table format "
        f"Do not include any tags like ```json or explanation as this output will be directly consumed in a Python program\n\n{profit_loss_section}"
    )

    OPTIONS = {"num_ctx": OLLAMA_CONTEXT}
    STREAM = False

    response = requests.post(
        url=f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": f"{OLLAMA_MODEL}",
            "messages": [{"role": "user", "content": LLM_summary_request}],
            "options": OPTIONS,
            "stream": STREAM,
        },
    )

    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "No summary received.")
    else:
        print(f"LLM request failed with status code {response.status_code}")
        return "Failed to get summary from LLM."


def extract_balance_sheet_result(
    html_data, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_CONTEXT
):
    soup = BeautifulSoup(html_data, "html.parser")
    # Look for the section with id 'shareholding' and class 'card card-large'
    balance_sheet_section = soup.find(
        "section", {"id": "balance-sheet", "class": "card card-large"}
    )

    LLM_summary_request = (
        f"You are a stock market expert. Prepare a detailed json of balance sheet group by the date as output for example balance_sheet_results->date-> Equity Capital,Reserves,etc. based on below HTML which contains this data in an html table format "
        f"Do not include any tags like ```json or explanation as this output will be directly consumed in a Python program\n\n{balance_sheet_section}"
    )

    OPTIONS = {"num_ctx": OLLAMA_CONTEXT}
    STREAM = False

    response = requests.post(
        url=f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": f"{OLLAMA_MODEL}",
            "messages": [{"role": "user", "content": LLM_summary_request}],
            "options": OPTIONS,
            "stream": STREAM,
        },
    )

    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "No summary received.")
    else:
        print(f"LLM request failed with status code {response.status_code}")
        return "Failed to get summary from LLM."


def extract_pros_cons_info(html_data, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_CONTEXT):
    soup = BeautifulSoup(html_data, "html.parser")
    # Look for the section with id 'shareholding' and class 'card card-large'
    pros_cons_section = soup.find(
        "section", {"id": "analysis", "class": "card card-large"}
    )

    LLM_summary_request = f"You are a stock market expert. Extract pros and cons from below HTML section and give output in json as stock_pros_cons->pros/cons->pros and cons. Do not include any tags like ```json or explanation as this output will be directly consumed in a Python program\n\n{pros_cons_section}"
    OPTIONS = {"num_ctx": OLLAMA_CONTEXT}
    STREAM = False

    response = requests.post(
        url=f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": f"{OLLAMA_MODEL}",
            "messages": [{"role": "user", "content": LLM_summary_request}],
            "options": OPTIONS,
            "stream": STREAM,
        },
    )

    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "No summary received.")
    else:
        print(f"LLM request failed with status code {response.status_code}")
        return "Failed to get summary from LLM."


def extract_shareholding_info(html_data):
    soup = BeautifulSoup(html_data, "html.parser")
    shareholding_info = []

    # Look for the section with id 'shareholding' and class 'card card-large'
    shareholding_section = soup.find(
        "section", {"id": "shareholding", "class": "card card-large"}
    )

    return shareholding_section


def extract_concall_info(html_data):
    soup = BeautifulSoup(html_data, "html.parser")
    concall_info = []

    # Look for the section with id 'documents' and class 'card card-large'
    concall_section = soup.find(
        "section", {"id": "documents", "class": "card card-large"}
    )

    return concall_section


def extract_text_from_pdf(pdf_url):
    headers = {"User-Agent": ua.random}
    response = requests.get(pdf_url, headers=headers)
    if response.status_code == 200:
        with open("temp_concall.pdf", "wb") as f:
            f.write(response.content)
        parsed_pdf = parser.from_file("temp_concall.pdf")
        return parsed_pdf.get("content", "").strip()
    else:
        print(
            f"Failed to download PDF from {pdf_url} with status code {response.status_code}"
        )
        return ""


def concall_section_summary_llm(
    concall_section, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_CONTEXT
):
    LLM_summary_request = (
        f"You are a stock market expert. Extract the most recent 1 concall dates and URLs from the below HTML code, "
        f"which contains the dates and URLs of concalls as href. Provide the output only in JSON format like "
        f'{{"concalls": [{{ "date": "Jan 2025", "URL": "https://www.asd.com/"}}]}}. '
        f"Do not include any tags like ```json or explanation as this output will be directly consumed in a Python program using json.loads. \n\n\n{concall_section}"
    )
    OPTIONS = {"num_ctx": OLLAMA_CONTEXT}
    STREAM = False

    response = requests.post(
        url=f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": f"{OLLAMA_MODEL}",
            "messages": [{"role": "user", "content": LLM_summary_request}],
            "options": OPTIONS,
            "stream": STREAM,
        },
    )

    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "No summary received.")
    else:
        print(f"LLM request failed with status code {response.status_code}")
        return "Failed to get summary from LLM."


def summarize_concall_with_llm(
    extracted_text, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_CONTEXT
):
    LLM_summary_request = (
        f"You are a stock market expert. Summarize the below concall transcript and extract the valuable data "
        f"from it which can later be used to understand the company's current and future valuation and strategy.\n\n{extracted_text}"
    )
    OPTIONS = {"num_ctx": OLLAMA_CONTEXT}
    STREAM = False

    response = requests.post(
        url=f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": f"{OLLAMA_MODEL}",
            "messages": [{"role": "user", "content": LLM_summary_request}],
            "options": OPTIONS,
            "stream": STREAM,
        },
    )

    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "No summary received.")
    else:
        print(f"LLM request failed with status code {response.status_code}")
        return "Failed to get summary from LLM."


def shareholding_summary_with_llm(
    shareholding_section, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_CONTEXT
):

    print(OLLAMA_BASE_URL)
    exit

    LLM_summary_request = (
        f"You are a stock market expert. Prepare a detailed json of holdings group by the date as output for example Shareholding_Pattern->quarterly/yearly->date->promoters,FIIs,DIIs,government,public,no_of_shareholders based on below HTML which contains the share holding pattern of a stock in a html table format "
        f"which can later be used to understand the company's current and future valuation and strategy. Do not include any tags like ```json or explanation as this output will be directly consumed in a Python program\n\n{shareholding_section}"
    )
    OPTIONS = {"num_ctx": OLLAMA_CONTEXT}
    STREAM = False

    response = requests.post(
        url=f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": f"{OLLAMA_MODEL}",
            "messages": [{"role": "user", "content": LLM_summary_request}],
            "options": OPTIONS,
            "stream": STREAM,
        },
    )

    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "No summary received.")
    else:
        print(f"LLM request failed with status code {response.status_code}")
        return "Failed to get summary from LLM."


def getTicker(company_name):
    """
    Fetches the ticker symbol for a given company name using Yahoo Finance API.

    :param company_name: Name of the company to search for.
    :return: Ticker symbol if found, otherwise None.
    """
    yfinance_url = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    )
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(
        url=yfinance_url, params=params, headers={"User-Agent": user_agent}
    )

    if res.status_code == 200:
        data = res.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            return data["quotes"][0]["symbol"]  # Extract first matching ticker

    return None  # Return None if no ticker is found


def fetch_stock_data(ticker, period="1y"):
    """
    Fetch historical stock data using yfinance.
    :param ticker: Stock ticker symbol (e.g., 'INFY.NS' for Infosys)
    :param period: Time period for which to fetch the data. Default is '1y' (last year)
    :return: DataFrame containing historical stock data and stock info
    """
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period=period)
    stock_info = stock.info
    return hist_data, stock_info


def calculate_moving_averages(data, short_window=50, long_window=200):
    """
    Calculate moving averages (short and long).
    :param data: DataFrame containing historical stock data
    :param short_window: Short window for the short-term moving average. Default is 50 days.
    :param long_window: Long window for the long-term moving average. Default is 200 days.
    :return: DataFrame with added moving averages
    """
    data["trend_sma_50"] = (
        data["Close"].rolling(window=short_window, min_periods=1).mean()
    )
    data["trend_sma_200"] = (
        data["Close"].rolling(window=long_window, min_periods=1).mean()
    )
    return data


def calculate_rsi(data, window=14):
    """
    Calculate the RSI (Relative Strength Index).
    :param data: DataFrame containing historical stock data
    :param window: Period over which to compute RSI. Default is 14 days.
    :return: DataFrame with added RSI column
    """
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data["momentum_rsi"] = rsi
    return data


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD (Moving Average Convergence Divergence).
    :param data: DataFrame containing historical stock data
    :param short_window: Short window for the short-term EMA. Default is 12 days.
    :param long_window: Long window for the long-term EMA. Default is 26 days.
    :param signal_window: Window over which to compute MACD Signal line. Default is 9 days.
    :return: DataFrame with added MACD, MACD_signal, and MACD_diff columns
    """
    ema_short = data["Close"].ewm(span=short_window, adjust=False).mean()
    ema_long = data["Close"].ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    macd_signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_diff = macd_line - macd_signal_line
    data["trend_macd"] = macd_line
    data["trend_macd_signal"] = macd_signal_line
    data["trend_macd_diff"] = macd_diff
    return data


def calculate_bollinger_bands(data, window=20):
    """
    Calculate Bollinger Bands.
    :param data: DataFrame containing historical stock data
    :param window: Period over which to compute Bollinger Bands. Default is 20 days.
    :return: DataFrame with added Bollinger Bands columns
    """
    sma = data["Close"].rolling(window=window, min_periods=1).mean()
    std_dev = data["Close"].rolling(window=window, min_periods=1).std()
    bollinger_hband = sma + (2 * std_dev)
    bollinger_lband = sma - (2 * std_dev)
    data["volatility_bbhi"] = bollinger_hband
    data["volatility_bbli"] = bollinger_lband
    return data


def identify_support_resistance(data, num_levels=3):
    """
    Identify support and resistance levels using local minima/maxima.
    :param data: DataFrame containing historical stock data
    :param num_levels: Number of support/resistance levels to find
    :return: Lists of support and resistance levels
    """
    # Find local maxima (resistance levels)
    peaks, _ = find_peaks(data["High"], prominence=0.1 * np.std(data["High"]))
    resistance_levels = data.iloc[peaks]["High"].values
    # Find local minima (support levels) by negating the 'Low' column
    valleys, _ = find_peaks(-data["Low"], prominence=0.1 * np.std(data["Low"]))
    support_levels = data.iloc[valleys]["Low"].values
    # Sort and get unique values
    resistance_levels = np.unique(np.sort(resistance_levels))[:num_levels]
    support_levels = np.unique(np.sort(support_levels))[:num_levels]
    return resistance_levels.tolist(), support_levels.tolist()


def analyze_stock_trend(data):
    """
    Analyze stock trend using price and volume changes over the last 15 days.
    :param data: DataFrame containing historical stock data
    :return: List of dictionaries with trading signals for the last 15 days
    """
    # Calculate price and volume changes
    data["Price Change"] = data["Close"].diff()
    data["Volume Change"] = data["Volume"].diff()
    # Trading Signals
    conditions = [
        (data["Price Change"] > 0) & (data["Volume Change"] > 0),  # Price ↑ & Volume ↑
        (data["Price Change"] < 0) & (data["Volume Change"] > 0),  # Price ↓ & Volume ↑
        (data["Price Change"] > 0) & (data["Volume Change"] < 0),  # Price ↑ & Volume ↓
        (data["Price Change"] < 0) & (data["Volume Change"] < 0),  # Price ↓ & Volume ↓
    ]
    signals = [
        "Bullish (Price up & Volume up - More buyers)",  # Price ↑ & Volume ↑
        "Bearish (Price down & Volume up - More sellers)",  # Price ↓ & Volume ↑
        "Weak Buying Pressure (Price up & Volume down - Trend may not sustain)",  # Price ↑ & Volume ↓
        "Weak Selling Pressure (Price down & Volume down - Downtrend may not continue)",  # Price ↓ & Volume ↓
    ]
    data["Signal"] = np.select(conditions, signals, default="Neutral / No Clear Signal")
    # Convert to list of dictionaries with today's and yesterday's data
    trend_analysis = [
        {
            "date": str(index.date()),
            "today_close_price": round(row["Close"], 2),
            "today_volume": int(row["Volume"]),
            "yesterday_close_price": (
                round(data["Close"].shift(1).loc[index], 2)
                if not pd.isna(data["Close"].shift(1).loc[index])
                else None
            ),
            "yesterday_volume": (
                int(data["Volume"].shift(1).loc[index])
                if not pd.isna(data["Volume"].shift(1).loc[index])
                else None
            ),
            "price_change": (
                round(row["Price Change"], 2)
                if not pd.isna(row["Price Change"])
                else None
            ),
            "volume_change": (
                int(row["Volume Change"]) if not pd.isna(row["Volume Change"]) else None
            ),
            "signal": row["Signal"],
        }
        for index, row in data.tail(30).iterrows()
    ]
    return trend_analysis


def sentiment_analysis(news):
    """
    Perform sentiment analysis on recent news articles.
    Args:
        news (list): List of news articles.
    Returns:
        dict: Sentiment analysis results including titles and sentiments.
    """
    if not news:
        return {
            "news_titles": [],
            "news_summaries": [],
            "news_sentiments": [],
            "avg_news_sentiment": 0.0,
        }

    news_titles = []
    news_summaries = []
    news_sentiments = []

    for article in news[:5]:  # Analyze the 5 most recent articles
        title = article.get("content", {}).get("title", "")
        summary = article.get("content", {}).get("summary", "")

        if not title and not summary:
            continue

        full_text = f"{title} {summary}"
        blob = TextBlob(full_text)
        sentiment = blob.sentiment.polarity

        news_titles.append(title)
        news_summaries.append(summary)
        news_sentiments.append(sentiment)

    avg_news_sentiment = (
        sum(news_sentiments) / len(news_sentiments) if news_sentiments else 0.0
    )

    return {
        "news_titles": news_titles,
        "news_summaries": news_summaries,
        "news_sentiments": news_sentiments,
        "avg_news_sentiment": avg_news_sentiment,
    }


def risk_assessment(stock_data, benchmark_data):
    """
    Perform risk assessment for a given stock.
    Args:
        stock_data (pd.DataFrame): Historical data of the stock.
        benchmark_data (pd.DataFrame): Historical data of the benchmark index.
    Returns:
        dict: Risk assessment results.
    """
    # Align data on common date index
    aligned_data = pd.concat(
        [
            stock_data["Close"].rename("Stock"),
            benchmark_data["Close"].rename("Benchmark"),
        ],
        axis=1,
    ).dropna()
    if aligned_data.empty:
        return {
            "beta": np.nan,
            "sharpe_ratio": np.nan,
            "value_at_risk_95": np.nan,
            "max_drawdown": np.nan,
            "volatility": np.nan,
        }
    # Calculate returns
    stock_returns = aligned_data["Stock"].pct_change().dropna()
    benchmark_returns = aligned_data["Benchmark"].pct_change().dropna()
    if stock_returns.empty or benchmark_returns.empty:
        return {
            "beta": np.nan,
            "sharpe_ratio": np.nan,
            "value_at_risk_95": np.nan,
            "max_drawdown": np.nan,
            "volatility": np.nan,
        }
    # Calculate beta
    covariance = np.cov(stock_returns, benchmark_returns)[0][1]
    benchmark_variance = np.var(benchmark_returns)
    if benchmark_variance == 0:
        beta = np.nan
    else:
        beta = covariance / benchmark_variance
    # Calculate Sharpe ratio
    risk_free_rate = 0.04  # Assume a higher risk-free rate for India, e.g., 4%
    excess_returns = stock_returns - risk_free_rate
    if excess_returns.std() == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    # Calculate Value at Risk (VaR) at 95% confidence level
    var_95 = np.percentile(stock_returns, 5)
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + stock_returns).cumprod()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    return {
        "beta": beta,
        "sharpe_ratio": sharpe_ratio,
        "value_at_risk_95": var_95,
        "max_drawdown": max_drawdown,
        "volatility": stock_returns.std() * np.sqrt(252),
    }


def competitor_analysis(ticker, num_competitors=3):
    """
    Perform competitor analysis for a given stock.
    Args:
        ticker (str): The stock ticker symbol.
        num_competitors (int): Number of top competitors to analyze.
    Returns:
        dict: Competitor analysis results.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    sector = info.get("sector")
    industry = info.get("industry")

    # Get competitors in the same sector (Note: yfinance does not provide direct competitor data)
    # For demonstration, we will simulate some competitors
    simulated_competitors = {
        "Information Technology": ["INFY.NS", "TCS.NS", "WIPRO.NS"],
        "Consumer Discretionary": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],
    }

    if sector in simulated_competitors:
        competitors = [
            comp for comp in simulated_competitors[sector] if comp != ticker
        ][:num_competitors]
    else:
        competitors = []

    competitor_data = []
    for comp in competitors:
        comp_stock = yf.Ticker(comp)
        comp_info = comp_stock.info
        competitor_data.append(
            {
                "ticker": comp,
                "name": comp_info.get("longName"),
                "market_cap": comp_info.get("marketCap"),
                "pe_ratio": comp_info.get("trailingPE"),
                "revenue_growth": comp_info.get("revenueGrowth"),
                "profit_margins": comp_info.get("profitMargins"),
            }
        )

    return {"sector": sector, "industry": industry, "competitors": competitor_data}


def get_fundamental_analysis(stock_info):
    """
    Retrieves and formats stock fundamental analysis data from the provided stock info.
    :param stock_info: Dictionary containing stock information.
    :return: Dictionary containing detailed stock fundamental analysis.
    """
    # Construct dictionary structure
    stock_data = {
        "BasicInfo": {
            key: stock_info.get(key, "N/A")
            for key in [
                "shortName",
                "sector",
                "industry",
                "marketCap",
                "currency",
                "exchange",
            ]
        },
        "Valuation": {
            key: stock_info.get(key, "N/A")
            for key in [
                "currentPrice",
                "previousClose",
                "fiftyTwoWeekHigh",
                "fiftyTwoWeekLow",
                "trailingPE",
                "forwardPE",
                "priceToBook",
                "enterpriseToRevenue",
            ]
        },
        "Financials": {
            key: stock_info.get(key, "N/A")
            for key in [
                "totalRevenue",
                "netIncomeToCommon",
                "ebitda",
                "trailingEps",
                "forwardEps",
                "freeCashflow",
                "operatingCashflow",
                "grossMargins",
                "operatingMargins",
                "profitMargins",
            ]
        },
        "RiskMetrics": {
            key: stock_info.get(key, "N/A")
            for key in [
                "beta",
                "debtToEquity",
                "returnOnAssets",
                "returnOnEquity",
                "quickRatio",
                "currentRatio",
            ]
        },
        "Dividends": {
            key: stock_info.get(key, "N/A")
            for key in ["dividendYield", "dividendRate", "exDividendDate"]
        },
        "AnalystRatings": {
            key: stock_info.get(key, "N/A")
            for key in [
                "recommendationKey",
                "targetMeanPrice",
                "targetHighPrice",
                "targetLowPrice",
                "numberOfAnalystOpinions",
            ]
        },
        "HistoricalData": [],
    }

    # Fetch historical market data (Last 5 Days)
    stock = yf.Ticker(stock_info["symbol"])
    hist_data = stock.history(period="5d")
    for index, row in hist_data.iterrows():
        stock_data["HistoricalData"].append(
            {
                "date": str(index.date()),
                "Open": round(row["Open"], 2),
                "High": round(row["High"], 2),
                "Low": round(row["Low"], 2),
                "Close": round(row["Close"], 2),
                "Volume": int(row["Volume"]),
            }
        )
    return stock_data


def prepare_overall_analysis(ticker):
    """
    Prepare overall analysis for a given stock ticker and return the result in JSON format.
    :param ticker: Stock ticker symbol (e.g., 'INFY.NS')
    :return: JSON string containing structured analysis results
    """

    # Check if the ticker is an Indian stock
    if not (ticker.endswith(".NS") or ticker.endswith(".BO")):
        raise Exception("I can only analyze Indian stocks as of now.")

    # Fetch historical data and stock info in one request
    hist_data, stock_info = fetch_stock_data(ticker, period="1y")
    if hist_data.empty:
        return json.dumps(
            {"Error": "Historical data for the stock is empty."}, indent=4
        )

    # Calculate moving averages using industry-recommended windows
    hist_data = calculate_moving_averages(hist_data)

    # Calculate RSI using a 14-day window
    hist_data = calculate_rsi(hist_data)

    # Calculate MACD with default windows (12, 26, 9 days)
    hist_data = calculate_macd(hist_data)

    # Calculate Bollinger Bands with a 20-day window
    hist_data = calculate_bollinger_bands(hist_data)

    # Identify support and resistance levels
    support_levels, resistance_levels = identify_support_resistance(hist_data)

    # Analyze stock trend using price and volume changes for the last 15 days
    trend_analysis = analyze_stock_trend(hist_data)

    # Get the latest values for important technical indicators
    latest_data = hist_data.iloc[-1]
    technical_analysis = {
        "current_price": latest_data["Close"],
        "sma_50": (
            latest_data["trend_sma_50"]
            if not pd.isna(latest_data["trend_sma_50"])
            else None
        ),
        "sma_200": (
            latest_data["trend_sma_200"]
            if not pd.isna(latest_data["trend_sma_200"])
            else None
        ),
        "rsi": (
            latest_data["momentum_rsi"]
            if not pd.isna(latest_data["momentum_rsi"])
            else None
        ),
        "macd": (
            latest_data["trend_macd_diff"]
            if not pd.isna(latest_data["trend_macd_diff"])
            else None
        ),
        "bollinger_hband": (
            latest_data["volatility_bbhi"]
            if not pd.isna(latest_data["volatility_bbhi"])
            else None
        ),
        "bollinger_lband": (
            latest_data["volatility_bbli"]
            if not pd.isna(latest_data["volatility_bbli"])
            else None
        ),
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "price_volume_analysis": trend_analysis,
    }

    # Fetch news articles for sentiment analysis
    stock = yf.Ticker(ticker)
    news = stock.news

    # Perform sentiment analysis
    sentiment_results = sentiment_analysis(news)

    # Fetch benchmark data (Nifty 50 Index)
    benchmark_data = yf.Ticker("^NSEI").history(period="1y")
    if benchmark_data.empty:
        return json.dumps(
            {"Error": "Historical data for the benchmark is empty."}, indent=4
        )

    # Perform risk assessment
    risk_assessment_results = risk_assessment(hist_data, benchmark_data)

    # Perform competitor analysis
    competitor_results = competitor_analysis(ticker, num_competitors=3)

    # Perform fundamental analysis
    fundamental_results = get_fundamental_analysis(stock_info)

    # Combine all results into a structured dictionary
    combined_results = {
        "StockSymbol": ticker,
        "StockDescription": {
            "longName": stock_info.get("longName", "N/A"),
            "sector": stock_info.get("sector", "N/A"),
            "industry": stock_info.get("industry", "N/A"),
            "marketCap": f"{stock_info.get('marketCap', 0):,}",
            "currency": stock_info.get("currency", "N/A"),
            "exchange": stock_info.get("exchange", "N/A"),
            "longBusinessSummary": stock_info.get("longBusinessSummary", "N/A"),
        },
        "TechnicalAnalysis": technical_analysis,
        "SentimentAnalysis": sentiment_results,
        "RiskAssessment": risk_assessment_results,
        "CompetitorAnalysis": competitor_results,
        "FundamentalAnalysis": fundamental_results,
    }

    # Convert to JSON format with descriptions
    final_output = {
        "StockSymbol": combined_results["StockSymbol"],
        "StockDescription": combined_results["StockDescription"],
        "Analyses": {
            "TechnicalAnalysis": {
                "Description": "Technical analysis of stock price and volume trends.",
                "Details": combined_results["TechnicalAnalysis"],
            },
            "SentimentAnalysis": {
                "Description": "Sentiment analysis based on recent news articles.",
                "Details": combined_results["SentimentAnalysis"],
            },
            "RiskAssessment": {
                "Description": "Risk assessment metrics including beta, Sharpe ratio, Value at Risk (VaR), and maximum drawdown.",
                "Details": combined_results["RiskAssessment"],
            },
            "CompetitorAnalysis": {
                "Description": "Competitor analysis based on sector and industry.",
                "Details": combined_results["CompetitorAnalysis"],
            },
            "FundamentalAnalysis": {
                "Description": "Detailed fundamental analysis including basic info, valuation metrics, financials, risk metrics, dividends, analyst ratings, and historical data.",
                "Details": combined_results["FundamentalAnalysis"],
            },
        },
    }

    # Convert to JSON string
    return json.dumps(final_output, indent=4)


class Filter:

    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        OLLAMA_BASE_URL: str = Field(
            default="http://172.17.0.1:11434", description="Ollama URL."
        )
        OLLAMA_MODEL: str = Field(default="qwen2.5:32b", description="Ollama Model.")
        OLLAMA_CONTEXT: int = Field(default=30000, description="Ollama Context.")

    class UserValves(BaseModel):
        OLLAMA_BASE_URL: str = Field(
            default="http://172.17.0.1:11434", description="Ollama URL."
        )
        OLLAMA_MODEL: str = Field(default="qwen2.5:32b", description="Ollama Model.")
        OLLAMA_CONTEXT: int = Field(default=30000, description="Ollama Context.")

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __event_emitter__: None,
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        print(f"inlet:{__name__}")
        print(f"inlet:body:{body}")
        print(f"inlet:user:{__user__}")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Thinking, let me understand your question...",
                        "done": False,
                    },
                }
            )

        # Extract the last message from the user
        conversation = body["messages"]
        user_last_message = conversation[-1]["content"]

        # user_last_message = None
        # for message in body["messages"]:
        #    user_last_message = message["content"]

        system_message = {
            "role": "system",
            "content": (
                "You are an expert stock market analyst who is specialized in stock analysis and recommendation."
            ),
        }

        prompt = "Go through the below conversation, understand user's query and let me know if user's last query require us to fetch stock information for analysis? if yes, then reply only with comma separated company names like company1,company2. If user's query does not require any stock information to be fetched then reply only with \"No\""

        user_message_with_prompt = f"{prompt}\n\n{conversation}"
        updated_user_message = {"role": "user", "content": user_message_with_prompt}
        new_messages = [system_message, updated_user_message]

        # Corrected dictionary access and parameter names
        OLLAMA_BASE_URL = min(
            __user__["valves"].OLLAMA_BASE_URL, self.valves.OLLAMA_BASE_URL
        )
        OLLAMA_MODEL = min(__user__["valves"].OLLAMA_MODEL, self.valves.OLLAMA_MODEL)
        OLLAMA_CONTEXT = min(
            __user__["valves"].OLLAMA_CONTEXT, self.valves.OLLAMA_CONTEXT
        )

        STREAM = False
        OPTIONS = {"num_ctx": OLLAMA_CONTEXT}  # Corrected dictionary format

        try:
            response = requests.post(
                url=f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": f"{OLLAMA_MODEL}",
                    "messages": new_messages,
                    "options": OPTIONS,
                    "stream": STREAM,
                },
            )

            response.raise_for_status()
            api_response = response.json()
            assistant_message = (
                api_response.get("message", {}).get("content", "").strip().lower()
            )

            print(assistant_message)

            if assistant_message != "no":
                try:

                    # Get company names from the assistant's message
                    companies = [name.strip() for name in assistant_message.split(",")]

                    all_company_data = ""

                    for company_name in companies:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Fetching ticker name for : {company_name} ...",
                                    "done": False,
                                },
                            }
                        )

                        ticker = getTicker(company_name)

                        if ticker:

                            print(f"Company: {company_name} -> Ticker: {ticker}")

                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Analyzing the stock : {ticker} ...",
                                        "done": False,
                                    },
                                }
                            )

                            company_data = prepare_overall_analysis(ticker)

                            # Get holding and concall data fromm screener.in

                            html_data = fetch_html(ticker)
                            concall_summary = {"concall_data": []}
                            concall_section = ""
                            concall_json = ""

                            if html_data:

                                # Extract pros and cons from screener.in
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Fetching and analyzing insights and details : {ticker} ...",
                                            "done": False,
                                        },
                                    }
                                )

                                pros_cons_info = extract_pros_cons_info(
                                    html_data,
                                    OLLAMA_BASE_URL,
                                    OLLAMA_MODEL,
                                    OLLAMA_CONTEXT,
                                )

                                # Extract querterly results
                                querterly_results_info = extract_quarterly_result(
                                    html_data,
                                    OLLAMA_BASE_URL,
                                    OLLAMA_MODEL,
                                    OLLAMA_CONTEXT,
                                )

                                # Extract profit and loss results
                                profit_loss_results = extract_profit_loss_result(
                                    html_data,
                                    OLLAMA_BASE_URL,
                                    OLLAMA_MODEL,
                                    OLLAMA_CONTEXT,
                                )

                                # Extract profit and loss results
                                balance_sheet_results = extract_balance_sheet_result(
                                    html_data,
                                    OLLAMA_BASE_URL,
                                    OLLAMA_MODEL,
                                    OLLAMA_CONTEXT,
                                )

                                # Extract shareholding details
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Analyzing the share holding and concall details of : {ticker} ...",
                                            "done": False,
                                        },
                                    }
                                )

                                shareholding_info = extract_shareholding_info(html_data)

                                shareholding_json = shareholding_summary_with_llm(
                                    shareholding_info,
                                    OLLAMA_BASE_URL,
                                    OLLAMA_MODEL,
                                    OLLAMA_CONTEXT,
                                )

                                # Extract concall details

                                concall_details = extract_concall_info(html_data)
                                concall_json = concall_section_summary_llm(
                                    concall_details,
                                    OLLAMA_BASE_URL,
                                    OLLAMA_MODEL,
                                    OLLAMA_CONTEXT,
                                )

                                # Load JSON data
                                data = json.loads(concall_json)

                                # Loop through each concall entry
                                for concall in data["concalls"]:
                                    date = concall["date"]
                                    url = concall["URL"]
                                    pdf_text = extract_text_from_pdf(
                                        url
                                    )  # Assuming this function extracts text from the PDF
                                    if pdf_text:
                                        summary = summarize_concall_with_llm(
                                            pdf_text,
                                            OLLAMA_BASE_URL,
                                            OLLAMA_MODEL,
                                            OLLAMA_CONTEXT,
                                        )

                                        concall_summary["concall_data"].append(
                                            {
                                                "date": date,
                                                "URL": url,
                                                "summary": summary,
                                            }
                                        )

                                # Convert the dictionary back to JSON with proper formatting
                                final_concall_summary = json.dumps(
                                    concall_summary, indent=4
                                )

                                # Print or save the final JSON output
                                print(final_concall_summary)

                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Share holding and concall details of : {ticker} is completed ...",
                                        "done": True,
                                    },
                                }
                            )

                            all_company_data += (
                                f"###### stock details in json for company name: {company_name}######\n\n"
                                f"{company_data}\n"
                                f"{shareholding_json}\n"
                                f"{pros_cons_info}\n"
                                f"{querterly_results_info}\n"
                                f"{profit_loss_results}\n"
                                f"{balance_sheet_results}\n"
                                f"{final_concall_summary}\n##################\n\n\n"
                            )

                        else:
                            print(f"No ticker found for company: {company_name}")
                            raise Exception(
                                f"No ticker found for company: {company_name}"
                            )
                            exit

                    print(all_company_data)
                    new_user_message = {
                        "role": "user",
                        "content": (
                            f"You are an expert stock market analyst who is specialized in stock analysis and recommendation. Below you will have user's question following which you will be provided with the stock's data between ################## parameters to study fundamental, technical, sentimental, risk assessment, competitor analysis, share holding trend, pros and cons feedback, quarterly results, profit and loss results, balance sheet, concall summary. You must go through each of these important stock parameters in details and carefully answer user's question as a stock market expert. Make sure you must answer only based on the data provided below. Do not answer if user's question can not be answered from below provided data. You are an expert, do your own analysis based on below provided data so don't rely only on \"AnalystRatings\" parameter. Here, its is important to note that you do not answer the question if response does not have the context.\n\n\nUser asked question :{user_last_message}\n\n"
                            f"Below is a company data for your detailed expert level analysis\n\n\n{all_company_data}\n\n\n"
                        ),
                    }

                    body["messages"][-1] = new_user_message

                except Exception as e:
                    print(f"Error executing date command: {e}")
                    raise Exception(e)

        except requests.RequestException as e:
            print(f"Request error: {e}")
            return {"error": str(e)}

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Processing completed...", "done": True},
            }
        )

        return body
