import streamlit as st
import pandas as pd
from pandas import to_datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import plotly.express as px
import xlwings as xw


st.write("""
# Forecasting Engine Application
""")

st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV input file](https://github.com/MinhTran1506/Forecasting-Engine/blob/main/Holts-Winter-data-input-VNHOLSC064.csv)
""")
# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload you input CSV file", type=["csv"])

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
    st.subheader("Errors")
    st.write("MAPE:", mape, "%")
    st.write("MAD:", mad)
    st.write("MSE:", mse)
    st.write("Optimized alpha:", round(optimized_model.params['smoothing_level'], 4))
    st.write("Optimized beta:", round(optimized_model.params['smoothing_trend'], 4))
    st.write("Optimized gamma:", round(optimized_model.params['smoothing_seasonal'], 4))

    # Plot actual vs forecasted values
    plt.rcParams["figure.figsize"] = (15, 7)
    plt.plot(data, label='Actual')
    plt.plot(fitted_model.fittedvalues, label='Fitted Values')
    plt.plot(range(len(data) - 1, len(data) + len(forecast) - 1), forecast, label='Forecast', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Holt-Winter Forecast')
    plt.xlabel('Periods')
    plt.ylabel('Sales')

    # Show the plot using st.pyplot()
    st.subheader("Holt-Winter Forecast Diagram")
    st.pyplot()
    
    #print(optimized_model.summary())

# Load data
#data = pd.Series([661503, 441668, 800233, 695703, 831934, 563977, 632920, 653983, 567768, 671143, 
#                  698414, 735658, 768786, 576410, 925364, 603491, 815072, 779434, 625540, 708970, 
#                  706063, 775610, 673040, 713563, 843126, 612435, 998286, 968521, 931580, 832860, 
#                  546894, 520231, 569827, 718404, 684232, 762439, 989438, 730242, 1133694, 1255191, 
#                  1108661, 1047170, 738503, 805819, 943491, 809612, 916650, 1030273, 904093])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Holts-Winter-data-input-VNHOLSC064.csv")

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


# Calculate optimized parameter for the Holt-Winter's model
model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12, 
                           initialization_method="estimated")
optimized_model = model.fit(optimized=True)
opt_alpha = float(optimized_model.params['smoothing_level'])  # Convert to float
opt_beta = float(optimized_model.params['smoothing_trend'])   # Convert to float
opt_gamma = float(optimized_model.params['smoothing_seasonal'])  # Convert to float

# Set default values for sliders
alpha_slider = st.sidebar.slider('Alpha:', min_value=0.0, max_value=1.0, step=0.01, value=opt_alpha)
beta_slider = st.sidebar.slider('Beta:', min_value=0.0, max_value=1.0, step=0.01, value=opt_beta)
gamma_slider = st.sidebar.slider('Gamma:', min_value=0.0, max_value=1.0, step=0.01, value=opt_gamma)
periods_slider = st.sidebar.slider('Periods:', min_value=1, max_value=len(data), step=1, value=36)


def user_input_features():
    data = {'Alpha': alpha_slider,
            'Beta': beta_slider,
            'Gamma': gamma_slider,
            'Periods': periods_slider,
        }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

example_df = pd.concat([input_df, df], axis=0)
example_df = example_df[:1] # Selects only the first row (the user input data)s

# Displays the user input features
st.subheader('User Input Features')

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(example_df)

# Call the forecasting function
holts_winter_forecast(alpha_slider, beta_slider, gamma_slider, periods_slider)



prophet = pd.DataFrame(df)
prophet = prophet.rename(columns={'Vol': 'y'})

prophet['ds'] = pd.date_range('2020-06-01', '2023-05-01', freq='MS')
prophet['ds']= to_datetime(prophet['ds'])
prophet = prophet[['ds','y']]

fig = px.line(prophet, x="ds", y="y", hover_data=['ds', 'y'])

# Show the plot
st.plotly_chart(fig)

prophet_model = Prophet(interval_width=0.95)
prophet_model.fit(prophet)

#prophet_forecast = prophet_model.make_future_dataframe(periods=36, freq='MS')
prophet_forecast = prophet_model.make_future_dataframe(periods=36,freq='MS')
prophet_forecast = prophet_model.predict(prophet_forecast)

#Calculate MAPE for the prophet forecast
test_prophet = prophet['y']
y = prophet_forecast['yhat']
mape_prophet = round(np.mean(np.abs((test_prophet[-12:] - y[:len(test_prophet)]) / test_prophet)) * 100, 2)

st.subheader("Prophet Forecast Diagram")
plt.figure(figsize=(30, 15))
prophet_model.plot(prophet_forecast, xlabel='Date', ylabel='Forecast')
plt.title('Prophet Forecast')

# Show the Prophet plot using st.pyplot()
st.pyplot()

st.write("MAPE: ", mape_prophet, "%")
fig = px.line(prophet_forecast, x="ds", y="yhat",
                 hover_data=['ds', 'yhat'])
# Show the plot
st.plotly_chart(fig)


if st.button('View data in Excel'):
    xw.view(prophet_forecast[['ds','yhat']])