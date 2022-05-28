# Import libraries
import datetime
# Access the web
import webbrowser
# Take HTML txt by request
import requests
# Analyse data
import pandas as pd
import numpy as np
# Helps to wait for data to download
import time

# General scraper


def scrape_tick_and_name(url):
    """This function take a URL, witch thought of as a root to take ticker and company name
    :param url: string
    :return: DataFrame
    """
    # Read the HTML as text split by rows
    html = requests.get(url).text.split('\n')

    # Pick the row where the data stored at
    rows = html[277].split(',')

    # Set up structure
    row = {'tickers': [], 'names': []}

    # This inner function lets you extract the word you want
    def extract_var(words):
        for i, c in enumerate(words):
            if c == ":":
                return words[i + 2:-1]

    for var in rows:
        if 'ticker' in var:
            row['tickers'].append(extract_var(var))
        if 'comp_name"' in var:
            row['names'].append(extract_var(var))
    return pd.DataFrame(row)


def get_links(url):
    """
    This function takes a head URL and "scarpe" the following URL to scrape data from
    :param url: string
    :return:  list of string
    """
    ## Read the HTML as text split by rows
    html = requests.get(url).text.split('\n')

    # Pick the row where the data stored at
    rows = html[277].split(',')

    # Set up the data
    links = []

    # This inner function lets you extract the word you want
    def extract_var(var1, var2):
        flag = False
        ind_i, ind_n = "", ""
        for c in var1:
          if c == ":" and not flag:
              flag = True
              continue
          elif c == '"' and flag:
              continue
          elif flag and c != '"':
              ind_i += c
        flag = False
        for c in var2:
          if c == ":" and not flag:
              flag = True
              continue
          elif c == '"' and flag:
              continue
          elif flag and c != '"':
              if c == " ":
                ind_n += "-"
              else:
                ind_n += c
        ind_n = ind_n.lower()
        link = "https://www.macrotrends.net/stocks/industry/" + ind_i + "/" + ind_n
        return link

    for ind, var in enumerate(rows):
        if 'zacks_x_ind_code' in var:
            the_link = extract_var(var, rows[ind + 1])
            if the_link != "":
                links.append(str(the_link))
    return links

# This function reads the variabels
def read_vars(vars):
    final_vars = []
    for var in vars:
        final_vars.append(var[9:19])
    return final_vars


# This function make the data
def read_original(original):
    """
    This function takes the original variable witch is the string of the correct variable of the HTML.
    Analyse it and return as dict
    :param original:  string
    :return: dict, list of keys to remember
    """
    final_data, keys = {}, []
    for data in original:
        # the one-on-one function to find the right data
        if " s: '" in data:
            keys.append(data[5:-1])
            final_data[keys[-1]] = {}
        elif (data[1:3] == '20' or data[1:3] == '19') and keys != []:
            final = ""
            for i in range(len(data)):
                if data[-(1 + i)] == ':':
                    break
                else:
                    final = data[-(1 + i)] + final
            final_data[keys[-1]][data[1:11]] = final

    return final_data, keys


# This function convert the vars to datetime objects
def convert_vars_to_datetime(vars):
    new_vars = []
    format_str = '%Y-%m-%d'  # The format
    for var in vars:
        new_vars.append(datetime.datetime.strptime(var, format_str))
    return new_vars


# This function takes a url and re shape it and take the data with the nake being gave in adition to the data
def scrape_a_company(url):
    # Set up endings
    endings = ["/income-statement?freq=Q", "/balance-sheet?freq=Q", "/cash-flow-statement?freq=Q",
               "/financial-ratios?freq=Q"]
    # Split the url
    url = url.split('/')
    # Take the name of company
    name = url[-2]
    # Set up data
    final_data = {}
    temp = ""
    for str_ in url[:-1]:
        temp += str_ + '/'
    url = str(temp)
    for end in endings:
        # Tempo url to "scarpe"
        temp = url + end
        # Take the html as text
        html = requests.get(temp).text.split('\n')
        # set up general keynames
        temp = end[1:-7]
        # Set up variables
        if end == '/financial-ratios?freq=Q':
            original = html[1134][20:].split(',')
            vars = html[1145].split(',')
        else:
            original = html[1142][20:].split(',')
            vars = html[1153].split(',')
        # Process the data
        vars = read_vars(vars[4::2])
        n = len(vars)
        data, keys = read_original(original)
        # save the data
        final_data[temp] = {'data': data, 'keys': keys, 'vars': vars}
    return name, final_data


# Make the data DataFrame
def transform_data_to_DF(name, final_data):
    """
    This function transform the data to DataFrame
    """
    final_ = []
    for key in final_data.keys():
        # Un pack the data
        keys, vars, data = final_data[key]['keys'], final_data[key]['vars'], final_data[key]['data']
        dt = {}
        for i in keys:
            temp = []
            for j in vars:
                var = data[i][j].strip('}').strip(';').strip(']').strip('"')
                try:
                    temp.append(float(var))
                except:
                    temp.append(var)

            dt[i] = temp
        df = pd.DataFrame(dt)
        df['date'] = vars
        final_.append([key, vars, df])
        final_df = final_[0][2].copy()
        for data in final_:
            df_ = data[2]
            final_df = final_df.merge(df_, left_on='date', right_on='date',)
    return final_df

def read_label_data(url, ticker, vars):
    """
    This function download the history of the stuck price make from it target data
    and return a dataframe of what we need
    :param url: string
    :param ticker: name string
    :param vars: name2 string
    :return: DataFrame
    """
    # Download the data we need
    webbrowser.open(url)
    # We need to wait for the download to finish or the program crush
    time.sleep(5)
    # Read the Data from my PC
    df = pd.read_csv(r"C:\Users\AyeletRB\Desktop\win_the_market\nasdaq Computer and Technology Sector\MacroTrends_Data_Download_" + ticker + ".csv", header=9)
    df = df[['date', 'close']]
    df['date'] = pd.to_datetime(df['date'])
    dict_for_df = {'date':[], 'target':[]}
    n = len(vars)
    vars = convert_vars_to_datetime(vars)

    # Analyse for target
    for i in range(n):
        if i == 0:
            bools_ = np.array(df['date'] >= vars[i])
        else:
            bools_ = np.array(df['date'] >= vars[i]) * np.array(df['date'] < vars[i - 1])
        df_temp = df[bools_]
        dict_for_df['date'].append(vars[i])
        dict_for_df['target'].append(np.array(df_temp['close']))
    return pd.DataFrame(dict_for_df)


def save_to_csv(name_, ticker, df):
    """
    This function save the data in the PC
    :param name_: string name of company
    :param ticker: id1 string
    :param df: the data DataFrame
    :return: none
    """

    # Specify the name of the excel file
    file_name = name_ + "_" + ticker + "_" + "dataset" + '.xlsx'

    # saving the excelsheet
    df.to_excel(file_name)

def full_read_ticker_name(ticker, name):
    """
    Full "Scrape" vy ticker and name
    :param ticker: string
    :param name: string
    :return: none
    """
    # Basic Url to make new
    basic_url = "https://www.macrotrends.net/stocks/charts/GOOG/alphabet/stock-price-history"
    # Transform to this company X_data and y_data (F:X->y)
    url_X = basic_url[0:42] + ticker + basic_url[46] + name +basic_url[55:]
    url_y = "https://www.macrotrends.net/assets/php/stock_price_history.php?t=AAPL"[:-4] + ticker
    # "Scrape" the company
    name, final_data = scrape_a_company(url_X)
    data_X = transform_data_to_DF(name, final_data)
    varss = data_X['date']
    html = requests.get(url_y).text.split('\n')
    url_y = html[485][33:-3]
    data_y = read_label_data(url_y, ticker, varss)
    # Unite data
    data_X['target'] = data_y['target']
    save_to_csv(name, ticker, data_X)



link = "https://www.macrotrends.net/stocks/industry/209/electronics---manufacturing-machinery"

dt = scrape_tick_and_name(link)
# un-pack
tickers = dt['tickers']
names = dt['names']
for i in range(len(tickers)):
    full_read_ticker_name(tickers[i], names[i])