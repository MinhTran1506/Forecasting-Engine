import pandas as pd
import numpy as np
from prophet import Prophet
import itertools
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import plotly.express as px

plt.ion()

# Define function for interactive forecasting with Holt-Winter's method
def holts_winter_forecast(alpha, beta, gamma, periods):
    # Fit the Winter-Holt's model
    model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12, 
                                 initialization_method="estimated")

    fitted_model = model.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
    optimized_model = model.fit(optimized = True)

    # Generate forecast
    forecast = fitted_model.forecast(periods)
    
    # Split data into train and test sets
    train = data.values[:-12]
    test = data.values[-12:]
    
    # Calculate errors
    mape = round(np.mean(np.abs((test - forecast[:len(test)]) / test)) * 100, 2)
    mad = round(mean_absolute_error(test, forecast[:len(test)]), 2)
    mse = round(mean_squared_error(test, forecast[:len(test)]), 2)
    
    # Print errors
    print("MAPE:", mape, "%")
    print("MAD:", mad)
    print("MSE:", mse)
    print("Optimized alpha:", round(optimized_model.params['smoothing_level'],4))
    print("Optimized beta:", round(optimized_model.params['smoothing_trend'],4))
    print("Optimized gamma:", round(optimized_model.params['smoothing_seasonal'],4))
    
    # Plot actual vs forecasted values
    plt.rcParams["figure.figsize"] = (15,7)
    plt.plot(data, label='Actual')
    plt.plot(fitted_model.fittedvalues, label='Fitted Values')
    plt.plot(range(len(data)-1, len(data)+len(forecast)-1), forecast, label='Forecast', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Holt-Winter Forecast')
    plt.xlabel('Periods')
    plt.ylabel('Sales')
    plt.show()
    
    #print(optimized_model.summary())

# Load data
#data = pd.Series([661503, 441668, 800233, 695703, 831934, 563977, 632920, 653983, 567768, 671143, 
#                  698414, 735658, 768786, 576410, 925364, 603491, 815072, 779434, 625540, 708970, 
#                  706063, 775610, 673040, 713563, 843126, 612435, 998286, 968521, 931580, 832860, 
#                  546894, 520231, 569827, 718404, 684232, 762439, 989438, 730242, 1133694, 1255191, 
#                  1108661, 1047170, 738503, 805819, 943491, 809612, 916650, 1030273, 904093])
df = pd.read_csv('Holts-Winter data input.csv')
array = []
for i in df['Vol']:
    array.append(i)
data = pd.Series(array)

#Calculate optimized parameter for the Hotls-Winter's model
model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12, 
                           initialization_method="estimated")
optimized_model = model.fit(optimized = True)
opt_alpha = optimized_model.params['smoothing_level']
opt_beta = optimized_model.params['smoothing_trend']
opt_gamma = optimized_model.params['smoothing_seasonal']

# Create the interactive sliders for alpha, beta, gamma, and periods
alpha_slider = widgets.FloatSlider(value=opt_alpha, min=0.0, max=1.0, step=0.01, description='Alpha:')
beta_slider = widgets.FloatSlider(value=opt_beta, min=0.0, max=1.0, step=0.01, description='Beta:')
gamma_slider = widgets.FloatSlider(value=opt_gamma, min=0.0, max=1.0, step=0.01, description='Gamma:')
periods_slider = widgets.IntSlider(value=36, min=1, max=len(data), step=1, description='Periods:')


# Call the winter_holts_forecast function with the interactive sliders as inputs#
interact(holts_winter_forecast, alpha=alpha_slider, beta=beta_slider, gamma=gamma_slider, periods=periods_slider)

prophet = pd.DataFrame(df)
prophet.head()
prophet = prophet.rename(columns={'Vol': 'y'})


from pandas import to_datetime

prophet['ds'] = pd.date_range('2020-06-01', '2023-05-01', freq='MS')
prophet['ds']= to_datetime(prophet['ds'])
prophet = prophet[['ds','y']]
prophet.head()

fig = px.line(prophet, x="ds", y="y",
                 hover_data=['ds', 'y'])
fig.show()



prophet_model = Prophet(interval_width=0.95)
prophet_model.fit(prophet)

#prophet_forecast = prophet_model.make_future_dataframe(periods=36, freq='MS')
prophet_forecast = prophet_model.make_future_dataframe(periods=36,freq='MS')
prophet_forecast = prophet_model.predict(prophet_forecast)

#Calculate MAPE for the prophet forecast
test_prophet = prophet['y']
y = prophet_forecast['yhat']
mape_prophet = round(np.mean(np.abs((test_prophet[-12:] - y[:len(test_prophet)]) / test_prophet)) * 100, 2)

plt.figure(figsize=(30, 15))
prophet_model.plot(prophet_forecast, xlabel = 'Date', ylabel = 'Forecast')
plt.title('Prophet Forecast')

print("MAPE: ",mape_prophet, "%")
fig = px.line(prophet_forecast, x="ds", y="yhat",
                 hover_data=['ds', 'yhat'])

# Show the plot
fig.show()

import xlwings as xw
xw.view(prophet_forecast[['ds','yhat']])