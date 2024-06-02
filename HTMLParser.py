from bs4 import BeautifulSoup
from pprint import pprint
import pickle

HTMLFile = open("Stock-Market-Neighbors/Most Active Stocks Today - Yahoo Finance.html", "r")

index = HTMLFile.read()

s = BeautifulSoup(index, 'html.parser')

stock_codes = [tag.text for tag in s.find_all("a", class_="Fw(600) C($linkColor)")]

with open("Stock Market Analyzer KNN/stock_codes.pkl", "wb") as f:
    pickle.dump(stock_codes, f)