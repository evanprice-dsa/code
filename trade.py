# imports
from coinbase.wallet.client import Client as cbc
import time
from datetime import datetime
from datetime import timedelta
import csv
import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
import talib
import polars as pl
import numpy as np
import pyarrow
import math
from neuralprophet import NeuralProphet
from prophet import Prophet
pd.set_option('display.max_columns', None)


class Trade:
    def __init__(self):  
        start_time = time.time()                            # track runtime
        # output file name
        self.wirePath = "/home/dev/code/tmp/" + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".txt"
        
        # variables
        self.include = set()                                # available assets
        self.data = dict()                                  # to store all the pandas dataframes - asset : dataframe
        self.staticPrices = dict()                          # to store all prices after modeling at moment of trade
        self.movingPercent = dict()                         # to track price movements after trade
        self.nn_predictRank = dict()                        # neuralprophet projected quartile assignments
        self.ts_predictRank = dict()                        # prophet projected quartile assignments
        self.actualRank = dict()                            # realized quartile assignments
        self.client = open('/home/dev/code/dat/api.txt', 'r').read().splitlines()
        self.output_buffer = []                             # buffer output until moment of trade
        
        # look for new and removed assets
        self.syncWallets()
        
        # multithread for network call function getHistoric() and then technical analysis function calcTA()
        with ThreadPoolExecutor(max_workers = 10) as executor:
            futures = [executor.submit(self.getHistoric, asset) for asset in self.include]
            for future in futures:
                result = future.result()
                if result is not None:
                    asset, df = future.result()
                    if df is not None:
                        self.calcTA(asset)
                    else:
                        self.output_buffer.append(f"Error\nNo data for {asset}\n\n") 
        
        # BEGIN NN 1/2
        # '''
        # parallel execution using joblib for predict()
        results = Parallel(n_jobs = -1)(delayed(self.nn_predict)(asset) for asset in self.include)
       
        # collect predictions
        predictions = []
        for result in results:
            if result is not None:
                predictions.append(result)

        # filter out none and nan values, and ensure all elements are dictionaries
        filtered_predictions = []
        for d in predictions:
            if isinstance(d, dict):
                filtered_dict = {k: v for k, v in d.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
                filtered_predictions.append(filtered_dict)
        
        # combine filtered dictionaries
        combined_dict = {k: v for d in filtered_predictions for k, v in d.items()}
       
        # sort the combined dictionary by values in descending order
        sorted_dict = dict(sorted(combined_dict.items(), key = lambda item: item[1], reverse = True))
        
        # store prices at trade time for future tracking 
        keys_list = list(sorted_dict.keys())
        self.staticPrices = self.tradePrice(keys_list)
        
        self.output_buffer.append(f"Top neural net picks\n{sorted_dict}\n\n")
        self.output_buffer.append(f"Prices at time of trade\n{self.staticPrices}\n\n")
        # '''
        # END NN 1/2
        
        # BEGIN TS 1/2
        # '''
        # parallel execution using joblib for predict()
        tresults = Parallel(n_jobs = -1)(delayed(self.ts_predict)(asset) for asset in keys_list)
       
        # collect predictions
        tpredictions = []
        for result in tresults:
            if result is not None:
                tpredictions.append(result)

        # filter out none and nan values, and ensure all elements are dictionaries
        tfiltered_predictions = []
        for d in tpredictions:
            if isinstance(d, dict):
                tfiltered_dict = {k: v for k, v in d.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
                tfiltered_predictions.append(tfiltered_dict)
        
        # combine filtered dictionaries
        tcombined_dict = {k: v for d in tfiltered_predictions for k, v in d.items()}
       
        # sort the combined dictionary by values in descending order
        tsorted_dict = dict(sorted(tcombined_dict.items(), key = lambda item: item[1], reverse = True))
        tkeys_list = list(tsorted_dict.keys())
        
        self.output_buffer.append(f"Top time series picks\n{tsorted_dict}\n\n")  
        # '''
        # END TS 1/2

        end_time = time.time()
        sec = end_time - start_time
        self.output_buffer.append(f"Modeling Runtime\n")
        self.output_buffer.append(f"{round(sec, 2)} seconds\n")
        self.output_buffer.append(f"{round((sec / 60), 2)} minutes\n\n")
        self.output(''.join(self.output_buffer))                    # only one write for output_buffer until modeling is complete

        # BEGIN NN 2/2
        # '''
        # divide predictions by quartile and assign value 1 to 4 with 1 being the highest
        rank = self.assign_quartiles_by_index(keys_list)            # keys_list in parallel mode
        for index, value in enumerate(keys_list):                   # keys_list in parallel mode
            self.nn_predictRank[value] = rank[index]
        self.output(f"Projected neural net quartile assignments\n{self.nn_predictRank}\n\n")
        # '''
        # END NN 2/2

        # BEGIN TS 2/2
        # '''
        # divide predictions by quartile and assign value 1 to 4 with 1 being the highest
        trank = self.assign_quartiles_by_index(tkeys_list)          
        for index, value in enumerate(tkeys_list):                  
            self.ts_predictRank[value] = trank[index]
        self.output(f"Projected time series quartile assignments\n{self.ts_predictRank}\n\n")
        # '''
        # END TS 2/2

        
        # '''
        # track price movement by percentage change increase for ~2 hours
        self.trackMovement(keys_list)                               # keys_list in parallel

        # final output csv style
        nn_matches = 0
        ts_matches = 0
        g1 = 0
        self.output(f"asset,nnpredict,tspredict,actual\n")
        for i in keys_list:
            self.output(f"{i},{self.nn_predictRank[i]},{self.ts_predictRank[i]},{self.actualRank[i]}\n")
            if self.nn_predictRank[i] == 1:
                g1 += 1
                if self.nn_predictRank[i] == self.actualRank[i]:
                    nn_matches += 1
                elif self.ts_predictRank[i] == self.actualRank[i]:
                    ts_matches += 1

        nn_lastNumber = round(float(nn_matches / g1), 2)
        nn_lastNumber *= 100
        ts_lastNumber = round(float(ts_matches / g1), 2)
        ts_lastNumber *= 100

        self.output(f"\nQuartile one prediction accuracy\n")
        self.output(f"NeuralProphet {nn_lastNumber}%\n")
        self.output(f"      Prophet {ts_lastNumber}%\n")
        # '''


    # streamline output
    def output(self, message):
        with open(self.wirePath, "a") as wire:
            wire.write(message)


    # match known assets with all available via api key and check asset availability changes
    def syncWallets(self):     
        # read from input file
        exclude = set()
        with open("/home/dev/code/dat/inp.txt", "r") as infile:
            reader = csv.reader(infile)
            for row in reader:
                if row[1] == "0":           
                    self.include.add(row[0])
                elif row[1] == "1":
                    exclude.add(row[0])
        
        try:
            cb_client = cbc(self.client[0], self.client[1])
            account = cb_client.get_accounts(limit=300)
        except Exception as e:
            self.output_buffer.append(f"Error\n{str(e)}\n\n")
            return
        
        # check for new/removed assets
        names = {wallet['name'].replace(" Wallet", "") for wallet in account.data}
        new_cryptos = names - self.include - exclude
        removed_cryptos = self.include - names 
        
        if new_cryptos:
            self.output_buffer.extend([f"### NEW CRYPTO -> {n} ###\n\n" for n in new_cryptos])
        if removed_cryptos:
            self.output_buffer.extend([f"### REMOVED CRYPTO -> {n} ###\n\n" for n in removed_cryptos])


    # live authenticated price for one asset
    def getPrice(self, asset):
        try:
            cb_client = cbc(self.client[0], self.client[1])
            currency_pair = f"{asset}-USD"
            price = cb_client.get_spot_price(currency_pair=currency_pair)
            return price['amount']
        except Exception as e:
            self.output_buffer.append(f"Error\n{asset}\n{e}\n\n")
            return None


    # get candlestick data for one asset
    def getHistoric(self, asset):
        attempts = 0                                                    # give it five tries
        while attempts < 5:
            attempts += 1
            try:
                assetUSD = asset + "-USD"
                url = f"https://api.pro.coinbase.com/products/{assetUSD}/candles"
            
                # send get request to the api
                response = requests.get(url, params = {'granularity': 900})
                response.raise_for_status()                             # exception for HTTP errors
                raw = response.json()
                raw = raw[::-1]
                df = pd.DataFrame(raw, columns = ['Date', 'Low', 'High', 'Open', 'Close', 'Volume'])
                df['Date'] = pd.to_datetime(df['Date'], unit = 's')     # make date readable  
                df.rename(columns = {'Date': 'ds'}, inplace = True)
                
                # log of percentage change is the target variable
                df['y'] = np.log(1 + df['Close'].pct_change())
                
                self.data[asset] = df
                
                # self.output_buffer.append(f"{asset}\n{self.data[asset]}\n\n")
                return asset, df
            
            except requests.exceptions.RequestException as e:
                self.output_buffer.append(f"Error\n{str(e)} on attempt {attempts} for {asset}\n\n")
                if attempts == 5:
                    return asset, None      # must return tuple

    
    # helper function for calcTA()
    def calculate_ema(self, polars_series, span):
        try:
            alpha = 2 / (span + 1)
            
            # initialize the ema with the first value
            ema_values = [polars_series[0]]
            for price in polars_series[1:]:
                # compute the next ema value based on the previous ema value
                next_ema = alpha * price + (1 - alpha) * ema_values[-1]
                ema_values.append(next_ema)
            return pl.Series(ema_values)
        except Exception as e:
            self.output_buffer.append(f"{e}\n\n")


    # calculate technical analysis indicators from the article
    # ma crosses, rsi, kdj, macd top and bottom structures, macd golden and dead crosses
    # trade signals -> 0 is neutral, 1 is buy and 2 is sell
    def calcTA(self, asset):
        try:
            # convert to polars for parallel processing and multithreading
            polarsDF = pl.DataFrame(self.data[asset], schema = ['ds', 'Low', 'High', 'Open', 'Close', 'Volume', 'y'])

            # 1/5 moving average crosses
            close_prices = polarsDF['Close'].to_numpy()                         # convert to numpy arrays for compatibility and performance
            short_ma = pl.Series(talib.SMA(close_prices, timeperiod = 5))       # 5 period moving average
            long_ma = pl.Series(talib.SMA(close_prices, timeperiod = 10))       # 10 period moving average
            polarsDF = polarsDF.with_columns([pl.Series('short_ma', short_ma), pl.Series('long_ma', long_ma)]) 
            
            ma_signal = np.where((polarsDF['short_ma'] > polarsDF['long_ma']) & (polarsDF['short_ma'].shift(1) <= polarsDF['long_ma'].shift(1)), 1, np.where((polarsDF['short_ma'] < polarsDF['long_ma']) & (polarsDF['short_ma'].shift(1) >= polarsDF['long_ma'].shift(1)), 2, 0))
            polarsDF = polarsDF.with_columns(pl.Series(name = 'ma_signal', values = ma_signal))

            # 2/5 rsi
            short_rsi = pl.Series(talib.RSI(close_prices, timeperiod = 5))
            long_rsi = pl.Series(talib.RSI(close_prices, timeperiod = 10))
            
            # signal generation based on rsi levels
            buy_signal = np.where((short_rsi < 50) & (short_rsi.shift(1) > long_rsi.shift(1)), 1, 0)
            sell_signal = np.where((short_rsi > 50) & (short_rsi.shift(1) < long_rsi.shift(1)), 2, 0)
            
            # combine buy and sell signals to create final rsi signals and add to dataframe
            rsi_signals = np.where(buy_signal == 1, 1, np.where(sell_signal == 2, 2, 0))
            polarsDF = polarsDF.with_columns([pl.Series('short_rsi', short_rsi), pl.Series('long_rsi', long_rsi),pl.Series('rsi_signal', rsi_signals)])

            # 3/5 kdj
            # calculate row stochastic value
            high_prices = polarsDF['High'].to_numpy()
            low_prices = polarsDF['Low'].to_numpy()
            close9 = talib.EMA(close_prices, timeperiod = 9)
            high9 = talib.MAX(high_prices, timeperiod = 9)      # highest price within 9 periods
            low9 = talib.MIN(low_prices, timeperiod = 9)        # lowest price within 9 periods
            rsv = ((close9 - low9) / (high9 - low9)) * 100
            
            # calculate k, d, j
            k = talib.EMA(rsv, timeperiod=3)
            d = talib.EMA(k, timeperiod=3)
            j = 3 * k - 2 * d
            
            # buy and sell signals
            buy_kdj = (k < 25) & (d < 25)
            sell_kdj = (k > 75) & (d > 75)

            # combine buy and sell signals
            kdj_signals = np.zeros_like(k, dtype=int)  # Initialize with zeros
            kdj_signals[buy_kdj] = 1  # Set buy signals
            kdj_signals[sell_kdj] = 2  # Set sell signals

            # add kdj signals to the polars dataframe
            polarsDF = polarsDF.with_columns([pl.Series('k', k), pl.Series('d', d), pl.Series('j', j), pl.Series('kdj_signal', kdj_signals)])

            # 4/5 macd top and bottom structures
            # calculate ema12, ema26, dif, dea, and macd
            ema12 = self.calculate_ema(close_prices, 12)
            ema26 = self.calculate_ema(close_prices, 26)
            dif = ema12 - ema26
            dea = self.calculate_ema(dif, 9)
            macd = (dif - dea) * 2

            # add the calculated columns
            polarsDF = polarsDF.with_columns([pl.Series('ema12', ema12), pl.Series('ema26', ema26), pl.Series('dif', dif), pl.Series('dea', dea), pl.Series('macd', macd)])
            
            # identify rising and falling waves
            rising_wave = (macd > 0) & (macd.shift(1) > 0)
            rising_wave_int = rising_wave.map_elements(lambda x: 1 if x else 0, return_dtype = pl.Int32)
            falling_wave = (macd < 0) & (macd.shift(1) < 0)
            falling_wave_int = falling_wave.map_elements(lambda x: 1 if x else 0, return_dtype = pl.Int32)

            # identify top and bottom structures
            # must convert to int after pandas
            close = polarsDF['Close']
            top_condition = ((close > close.shift())        # current close price is higher than previous close price
                & (macd > macd.shift())                     # current macd is higher than previous macd
                & (macd < macd.shift(2)))                   # current macd is lower than macd two intervals ago

            bottom_condition = ((close < close.shift())     # current close price is lower than previous close price
                & (macd < macd.shift())                     # current macd is lower than previous macd
                & (macd > macd.shift(2)))                   # current macd is higher than macd two intervals ago

            polarsDF = polarsDF.with_columns([pl.Series('rising_wave', rising_wave_int), pl.Series('falling_wave', falling_wave_int), pl.Series('top_structures', top_condition), pl.Series('bottom_structures', bottom_condition)]) 
            
            # golden and dead cross signals
            golden_cross = [0] * len(polarsDF)  # Pre-fill with 0s
            dead_cross = [0] * len(polarsDF)    # Pre-fill with 0s

            # convert dif and dea to numpy arrays for iteration
            dif_array = dif.to_numpy()
            dea_array = dea.to_numpy()

            # iterate over the data starting from the second element
            for i in range(1, len(dif_array)):
                # Check for golden cross
                if (dif_array[i] > dea_array[i] and dif_array[i-1] < dea_array[i-1] and dif_array[i] > 0 and dea_array[i] > 0):
                    golden_cross[i] = 1

                # Check for dead cross
                if (dif_array[i] < dea_array[i] and dif_array[i-1] > dea_array[i-1] and dif_array[i] < 0 and dea_array[i] < 0):
                    dead_cross[i] = 2

            # convert lists to polars Series
            golden_cross_series = pl.Series(golden_cross)
            dead_cross_series = pl.Series(dead_cross)   

            # add columns to polarsDF
            polarsDF = polarsDF.with_columns([pl.Series('golden', golden_cross_series), pl.Series('dead', dead_cross_series)
])

            # polars to pandas
            polars_df = polarsDF.to_pandas().dropna()

            # convert boolean columns to integers to compensate for polars to pandas data conversion issues
            boolean_columns = ['top_structures', 'bottom_structures', 'rising_wave', 'falling_wave', 'golden', 'dead']
            for col in boolean_columns:
                polars_df[col] = polars_df[col].astype(int)          
            
            self.data[asset] = polars_df

        except Exception as e:
            self.output_buffer.append(f"{asset}\n{e}\n\n")

    
    # neural network modeling 
    def nn_predict(self, asset):
        try:
            # last timestamp
            last_observed_time = self.data[asset]['ds'].max()
        
            # 15 minutes, 30, 45, 60, 75, 90, 105 from the last ... but 15 minutes data is to calculate y value for 30
            future_times = [last_observed_time + timedelta(minutes = 15 * i) for i in range(7)]

            # main model
            nmodel = NeuralProphet()

            # create dataframe with the future timestamps and the last observed 'y' value
            future_dates = pd.DataFrame({'ds': future_times})
            np_fut_dates = nmodel.make_future_dataframe(self.data[asset]) 
            np_fut_dates = pd.concat([np_fut_dates, future_dates], ignore_index = True)
            np_fut_dates = np_fut_dates.drop(index = 0)
            np_fut_dates['y'] = self.data[asset]['y'].iloc[-1]           
            
            # features to be used
            features = ['Low', 'High', 'Open', 'Close', 'Volume', 'short_ma', 'long_ma', 'ma_signal', 'short_rsi', 'long_rsi', 'rsi_signal', 'k', 'd', 'j', 'kdj_signal', 'ema12', 'ema26', 'dif', 'dea', 'macd', 'rising_wave', 'falling_wave', 'top_structures', 'bottom_structures', 'golden', 'dead']
            
            # neuralprophet automatically removes columns with only one unique value
            features_to_drop = []
            for f in features:
                if self.data[asset][f].nunique() == 1:
                    features_to_drop.append(f)
                    del np_fut_dates[f]

            features = [f for f in features if f not in features_to_drop]

            # fill in the regressor values for the future timestamps
            for idx, future_time in np_fut_dates.iterrows():
                for f in features:
                    np_fut_dates.loc[idx, f] = self.data[asset][f].iloc[-1]
            
            # add features to the model
            for f in features:
                nmodel.add_lagged_regressor(f)

            # fit the primary model and predict for the historical data
            nmodel.fit(self.data[asset][['ds', 'y'] + features])
            historical_predictions = nmodel.predict(self.data[asset][['ds', 'y'] + features])

            # calculate residuals
            residuals = self.data[asset]['y'] - historical_predictions['yhat1']

            # prepare data for residuals model
            residuals_data = self.data[asset].copy()
            residuals_data['y'] = residuals

            # initialize and fit the residuals model
            rmodel = NeuralProphet()
            for f in features:
                rmodel.add_lagged_regressor(f)
            rmodel.fit(residuals_data[['ds', 'y'] + features])
            
            # predict future values for both models and combine
            primary_predictions = nmodel.predict(np_fut_dates)
            residuals_predictions = rmodel.predict(np_fut_dates)
            final_predictions = primary_predictions['yhat1'] + residuals_predictions['yhat1']

            # return as a dictionary
            return {asset : round(final_predictions.max(), 4)}

        except Exception as e:
            self.output_buffer.append(f"{asset}\n{e}\n\n")
            return asset, None

        
    # time series modeling 
    def ts_predict(self, asset):
        try:
            # last timestamp
            last_observed_time = self.data[asset]['ds'].max()
        
            # 15 minutes, 30, 45, 60, 75, 90, 105 from the last ... but 15 minutes data is to calculate y value for 30
            future_times = [last_observed_time + timedelta(minutes = 15 * i) for i in range(7)]

            # main model
            tmodel = Prophet()

            # create dataframe with the future timestamps and the last observed 'y' value
            future_dates = pd.DataFrame({'ds': future_times})
            future_dates['y'] = self.data[asset]['y'].iloc[-1]           
            
            # features to be used
            features = ['Low', 'High', 'Open', 'Close', 'Volume', 'short_ma', 'long_ma', 'ma_signal', 'short_rsi', 'long_rsi', 'rsi_signal', 'k', 'd', 'j', 'kdj_signal', 'ema12', 'ema26', 'dif', 'dea', 'macd', 'rising_wave', 'falling_wave', 'top_structures', 'bottom_structures', 'golden', 'dead']

            # fill in the regressor values for the future timestamp
            for f in features:
                future_dates[f] = self.data[asset][f].iloc[-1]    

            # removes column with only one unique value
            features_to_drop = []
            for f in features:
                if self.data[asset][f].nunique() == 1:
                    features_to_drop.append(f)
                    del future_dates[f]
            
            features = [f for f in features if f not in features_to_drop]

            # fill in the regressor values for the future timestamps
            for idx, future_time in future_dates.iterrows():
                for f in features:
                    future_dates.loc[idx, f] = self.data[asset][f].iloc[-1]            

            # add features to the model
            for f in features:
                tmodel.add_regressor(f)

            # fit the primary model and predict for the historical data
            tmodel.fit(self.data[asset][['ds', 'y'] + features])
            historical_predictions = tmodel.predict(self.data[asset][['ds', 'y'] + features])

            # calculate residuals
            residuals = self.data[asset]['y'] - historical_predictions['yhat']

            # prepare data for residuals model
            residuals_data = self.data[asset].copy()
            residuals_data['y'] = residuals

            # initialize and fit the residuals model
            rmodel = Prophet()
            for f in features:
                rmodel.add_regressor(f)
            rmodel.fit(residuals_data[['ds', 'y'] + features])
            
            # predict future values for both models and combine
            primary_predictions = tmodel.predict(future_dates)
            residuals_predictions = rmodel.predict(future_dates)
            final_predictions = primary_predictions['yhat'] + residuals_predictions['yhat']
            
            # return as a dictionary
            return {asset : round(final_predictions.max(), 4)}

        except Exception as e:
            self.output_buffer.append(f"{asset}\n{e}\n\n")
            return asset, None


    # input a list of assets and return the price at the trade moment
    def tradePrice(self, assets):
        priceCapture = {}
        # multithread for network calls
        with ThreadPoolExecutor(max_workers = 10) as executor:
            # submit all tasks concurrently
            futures = {executor.submit(self.getPrice, asset): asset for asset in assets}

            # iterate through completed futures
            for future in futures:
                asset = futures[future]             # get the asset corresponding to the completed future
                try:
                    result = future.result()        # get the result of the completed future
                    priceCapture[asset] = result    # store the result in the dictionary
                except Exception as e:
                    self.output_buffer.append(f"Error fetching price for {asset}: {e}\n\n")
   
        return priceCapture

    
    # input a list and return quartile index rank 1 to 4 with 1 being the highest
    def assign_quartiles_by_index(self, data):
        n = len(data)
       
        # determine the size of each quartile and the remainder
        base_size = n // 4
        remainder = n % 4

        # list to hold the quartile sizes
        quartile_sizes = [base_size + (1 if i < remainder else 0) for i in range(4)]
        
        # function to determine the quartile based on index
        def get_quartile(index):
            accumulated_size = 0
            for i, size in enumerate(quartile_sizes):
                accumulated_size += size
                if index < accumulated_size:
                    return i + 1                   

        # assign quartile values based on index and return list
        return [get_quartile(i) for i in range(n)]


    # track the highest percentage change increase over ~2 hours and generate quartile list
    def trackMovement(self, assets):
        self.movingPercent = {asset: 0 for asset in assets}

        # helper function
        def percentage_change(old_value, new_value):
            # Calculate the percentage change
            change = ((new_value - old_value) / old_value) * 100
            return round(change, 4)     
        
        # below seven minute modeling equals 113 minutes of tracking time to remain below two hour total runtime
        # 113 minutes / 30 seconds = 226 iterations
        for i in range(226): 
            # pause
            time.sleep(30)
            
            with ThreadPoolExecutor(max_workers = 10) as executor:
                tempDict = dict()
                # submit all tasks concurrently
                futures = {executor.submit(self.getPrice, asset): asset for asset in assets}

                # iterate through completed futures
                for future in futures:
                    asset = futures[future]             # get the asset corresponding to the completed future
                    try:
                        result = future.result()        # get the result of the completed future
                        tempDict[asset] = result        # store the result in the dictionary
                        
                        change = percentage_change(float(self.staticPrices[asset]), float(tempDict[asset]))
                        
                        # check for increase
                        if change > self.movingPercent[asset]:
                            self.movingPercent[asset] = change

                    except Exception as e:
                        self.output_buffer.append(f"Error fetching price for {asset}: {e}\n\n")
            
            sorted_dict = dict(sorted(self.movingPercent.items(), key = lambda item: item[1], reverse = True))
            self.output(f"Iteration {int(i + 1)}\n{sorted_dict}\n\n")

        # convert dictionary keys to a list
        keys_list = list(sorted_dict.keys())
        
        # divide predictions by quartile and assign value 1 to 4 with 1 being the highest
        rank = self.assign_quartiles_by_index(keys_list)
        for index, value in enumerate(keys_list):
            self.actualRank[value] = rank[index]
        
        self.actualRank = dict(sorted(self.actualRank.items(), key = lambda item: item[1]))
        self.output(f"Actual quartile assignments\n{self.actualRank}\n\n")
     

# void main
_new = Trade()
