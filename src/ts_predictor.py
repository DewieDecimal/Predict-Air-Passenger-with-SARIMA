# Import and set up basic packages
import numpy as np
import pandas as pd

# Import tools
import joblib
import statsmodels.api as sm

# Import models
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Create a wrangle function
def wrangle(filepath):
    # Read data from csv file
    df = pd.read_csv(filepath, index_col='Month')

    #Tranform the month column 
    df.index = pd.DatetimeIndex(df.index)

    return df


def Air_Passenger_Predictor(modelname, start_date, end_date, dynamic=False, return_series=False):

    """
    Generate predictions using a SARIMA model.
    
    Parameters:
    - model: Fitted SARIMA model.
    - start_date: Start date for the prediction period.
    - end_date: End date for the prediction period.
    - dynamic: Whether to use dynamic forecasting (default: False).
    - return_series: Whether the output will be returned as a Series or just an array of values.
    
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
    # When you set dynamic=True, the model continuously predicts one-step ahead (t+1) and then for the 2nd step ahead (t+2) prediction, it appends predicted value (t+1) to data, re-fits model on new expanded data then makes 2nd step ahead forecast. This is called out-of-sample prediction.

    # When you set dynamic=False, the model sequentially predicts one-step-ahead using the true value from previous time step instead of using predicted value. This is called in-sample prediction.

    # Extract and return predicted values
    predicted_values = predictions.values
    
    if return_series:
        return predictions
    else:
        return predicted_values

