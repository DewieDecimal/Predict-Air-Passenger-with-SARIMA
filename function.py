# Import and set up basic packages
import numpy as np

import pandas as pd
pd.options.mode.copy_on_write = True

import matplotlib.pyplot as plt


# Import tools
import joblib
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error

# Import models
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX




# Create a wrangle function
def wrangle(filepath):
    # Read data from csv file
    df = pd.read_csv(filepath, index_col='Month')

    #Tranform the month column 
    df.index = pd.DatetimeIndex(df.index)

    return df


# Create a function to decompose time-series dataframe
def decomposer(ts_dataframe):
    decompose = seasonal_decompose(ts_dataframe, model='additive')
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 7))
    decompose.observed.plot(ylabel='Observed', ax=ax[0])
    decompose.trend.plot(ylabel='Trend', ax=ax[1])
    decompose.seasonal.plot(ylabel='Seasonal', ax=ax[2])
    decompose.resid.plot(ylabel='Resid', ax=ax[3])
    plt.xlabel('Date')



def SARIMA_predict(modelname, start_date, end_date, dynamic=False, return_series=False):

    """
    Generate predictions using a SARIMA model.
    
    Parameters:
    - model: Fitted SARIMA model.
    - start_date: Start date for the prediction period.
    - end_date: End date for the prediction period.
    - dynamic: Whether to use dynamic forecasting (default: False).
    
    Returns:
    - predictions: Predicted values.
    """

    model = joblib.load(modelname)

    # Generate the date range for prediction
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Generate predictions
    if dynamic:
        predictions = model.predict(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), dynamic=True)
    else:
        predictions = model.predict(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), dynamic=False)
    
    # Extract and return predicted values
    predicted_values = predictions.values
    
    if return_series:
        return predictions
    else:
        return predicted_values

