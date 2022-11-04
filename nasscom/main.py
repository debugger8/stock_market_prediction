# import streamlit for web_app 
from json import load
import streamlit as st
# #import date for the time_period
from datetime import date

# #import yahoo_finance to fetch data
import yfinance as yf


# #import graph as objects from plotly to visualise
from plotly import graph_objs as go

# #for working with csv
import csv

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import rcParams
from prepare_data import DataLoader
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os

# ---------- load inpputs and models ---------- #
def main():
    r=open('stock.csv','r')
    reader=csv.reader(r)

    people=[]

    for row in reader:
        people.append(row)
    
    stock_list=[]  
    for item in people:
        stock_list.append(item[0])

    #put the start date and current date
    START = "1990-01-01"
    TODAY=date.today().strftime('%Y-%m-%d')

    #title of web_app
    st.title("Stock Market Prediction Application")


    # #dataset selection
    selected_stocks = st.selectbox("Select Dataset for Prediction", stock_list) #selectbox will assign a value to that variable
    selected_stocks='BTC'   
    # #load stock data
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace = True)
        return data

    data_load_state = st.text("Load data...")

    data_stocks = load_data(selected_stocks)
    data_load_state.text("Results are...")

    # #calling raw data
    st.subheader('Raw data')
    st.write(data_stocks)
   
    with open('inputs.json') as f:
        inputs = json.load(f)
    months = inputs['months']
    model_name = inputs['model_name']
    models = inputs['models']
    crash_threshold = inputs['crash_threshold']
    n_lookback = inputs['n_days_lookback']
    n_plot = inputs['n_days_plot']
    
    # # ---------- load data ---------- #
    data = DataLoader(data_stocks)


    dataset_revised, crashes = data.get_data_revised(data_stocks,[crash_threshold])

    dfs_x, dfs_y = data.get_dfs_xy_predict(months=months)
    X, _, _, _ = data.get_train_test(dfs_x, dfs_y, data_stocks, test_data=None)
    os.chdir('..')
    
    # ---------- make predictions ---------- #
    y_pred_weighted_all = []
    for month, model in zip(months, models):
        model = pickle.load(open(model, 'rb'))
        y_pred_bin = model.predict(X).astype(int)
        y_pred_weighted = []
        for i in range(-n_plot, -1):
            y_pred_bin_ = y_pred_bin[:i] 
            y_pred_weighted.append(np.dot(np.linspace(0,1,21) / \
                    sum(np.linspace(0, 1, n_lookback)), y_pred_bin_[-n_lookback:]))
        y_pred_weighted.append(np.dot(np.linspace(0, 1, n_lookback) / \
                    sum(np.linspace(0, 1, n_lookback)), y_pred_bin[-n_lookback:]))
        y_pred_weighted_all.append(y_pred_weighted)

    # ---------- print and plot results ---------- #
    df = dataset_revised[0].iloc[-n_plot:, :]
    df['y_pred_weighted_1m'] = y_pred_weighted_all[0]
    df['y_pred_weighted_3m'] = y_pred_weighted_all[1]
    df['y_pred_weighted_6m'] = y_pred_weighted_all[2]
    last_date = str(df.index[-1])[:10]

    print(str(data_stocks) + ' crash prediction ' + str(model_name) + ' model as of '\
          + str(last_date))
    print('probabilities as weighted average of binary predictions over last '\
          + str(n_lookback) + str(' days'))
    print('* crash within 6 months: ' + str(np.round(100 \
            * df['y_pred_weighted_6m'][-1], 2)) + '%')
    print('* crash within 3 months: ' + str(np.round(100 \
            * df['y_pred_weighted_3m'][-1], 2)) + '%')
    print('* crash within 1 month:  ' + str(np.round(100 \
            * df['y_pred_weighted_1m'][-1], 2)) + '%')

    plt.style.use('seaborn-darkgrid')
    rcParams['figure.figsize'] = 10, 6
    rcParams.update({'font.size': 12})
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 1])
    plt.subplot(gs[0])
    plt.plot(df['price'], color='blue')
    plt.ylabel(str(selected_stocks) + ' index')
    plt.title(str(selected_stocks) + ' crash prediction ' + str(model_name) + ' ' \
              + str(last_date))
    plt.xticks([])
    plt.subplot(gs[1])
    plt.plot(df['y_pred_weighted_6m'], color='salmon')
    plt.plot(df['y_pred_weighted_3m'], color='red')
    plt.plot(df['y_pred_weighted_1m'], color='brown')
    plt.ylabel('crash probability')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45)
    plt.legend(['crash in 6 months', 'crash in 3 months', 'crash in 1 month'])
    plt.show()
    
    
    st.write('probabilities as weighted average of binary predictions over last ' + str(n_lookback) + str(' days'))
    st.write('* crash within 6 months: ' + str(np.round(100 * df['y_pred_weighted_6m'][-1], 2)) + '%')
    st.write('* crash within 3 months: ' + str(np.round(100 * df['y_pred_weighted_3m'][-1], 2)) + '%')
    st.write('* crash within 1 month:  ' + str(np.round(100 * df['y_pred_weighted_1m'][-1], 2)) + '%')
    
    
if __name__ == '__main__':
    main()