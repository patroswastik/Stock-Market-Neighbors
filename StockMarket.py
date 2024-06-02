import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from pprint import pprint
from sklearn.metrics.pairwise import euclidean_distances
import pickle

def ranking_algorithm(X: pd.DataFrame, stock_dataset: pd.DataFrame, query_str: str, top_k=3):
    scores = dict()
    for stock in stock_dataset.index:
        if "_scaled" in stock:
            scores[stock] = euclidean_distances(stock_dataset.loc[[stock]], X.loc[[f"{query_str}_scaled"]])[0, 0]
    ranking = list(sorted(scores.items(), key=lambda x : x[1]))[:top_k]
    plt.plot(X.loc[f"{query_str}_scaled"].T, "-r", label=f"Query: {query_str}_scaled", linewidth=2)
    for stock in ranking:
        plt.plot(stock_dataset.loc[stock[0]].T, label=f"{stock[0]}: {stock[1]:0.2f}")
    plt.legend(loc="upper left")
    plt.show()

    pprint(ranking)

def stock_dataframe(Close, Open, stock=None):
    mmscaler = MinMaxScaler()
    stock_data = np.array((Close[stock] + Open[stock]) / 2) if stock is not None else np.array((Close + Open) / 2)
    scaled_stock_data = np.array(mmscaler.fit_transform(stock_data.reshape(-1, 1))).reshape(stock_data.shape)
    return stock_data, scaled_stock_data

def stock_data(period="6mo"):
    stock_codes = None
    with open("Stock Market Analyzer KNN/stock_codes.pkl", "rb") as f:
        stock_codes = pickle.load(f)
    multi_data = yf.download(stock_codes, period=period)
    mean_datas = dict()
    mean_index = multi_data.index

    column_names = []

    for stock in stock_codes:
        # mmscaler = MinMaxScaler()
        # stock_data = np.array((multi_data.Close[stock] + multi_data.Open[stock]) / 2)
        # scaled_stock_data = np.array(mmscaler.fit_transform(stock_data.reshape(-1, 1))).reshape(stock_data.shape)
        stock_data, scaled_stock_data = stock_dataframe(multi_data.Close, multi_data.Open, stock)
        mean_datas[stock] = stock_data
        mean_datas[f"{stock}_scaled"] = scaled_stock_data
        column_names += [stock, f"{stock}_scaled"]


    mean_stock_data = pd.DataFrame(data=mean_datas, columns=column_names, index=mean_index)
    # for stock in stock_codes:
    #     plt.plot(mean_stock_data[f"{stock}_scaled"], label=f"{stock}_scaled")
    # plt.show()
    return mean_stock_data.T

def stock_query(query: str, period="6mo"):
    query_data = yf.download(query, period=period)
    mean_index = query_data.index
    # mmscaler = MinMaxScaler()
    # stock_data = np.array((query_data.Close + query_data.Open) / 2)
    # scaled_stock_data = np.array(mmscaler.fit_transform(stock_data.reshape(-1, 1))).reshape(stock_data.shape)
    stock_data, scaled_stock_data = stock_dataframe(query_data.Close, query_data.Open)
    mean_data = {
        query: stock_data,
        f"{query}_scaled": scaled_stock_data
    }
    query_df = pd.DataFrame(mean_data, columns=[query, f"{query}_scaled"], index=mean_index)
    # query_df[f"{query}_scaled"].plot()
    # plt.show()
    return query_df.T


if __name__ == "__main__":
    query_str = "DOX"
    query_period = "3mo"
    period = "3mo"
    stock_df = stock_data(period)
    print(stock_df.head())
    query = stock_query(query_str, query_period)
    print(query.head())
    ranking_algorithm(query, stock_df, query_str, 1)