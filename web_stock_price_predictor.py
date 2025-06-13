# import streamlit as st
# import pandas as pd
# import numpy as np
# from keras.models import load_model
# import matplotlib.pyplot as plt
# import yfinance as yf

# st.title("Stock Price Predictor App")

# stock = st.text_input("Enter the Stock ID", "GOOG")

# from datetime import datetime
# end = datetime.now()
# start = datetime(end.year-20,end.month,end.day)

# google_data = yf.download(stock, start, end)

# model = load_model("Latest_stock_price_model.keras")
# st.subheader("Stock Data")
# st.write(google_data)

# splitting_len = int(len(google_data)*0.7)
# x_test = pd.DataFrame(google_data.Close[splitting_len:])

# def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
#     fig = plt.figure(figsize=figsize)
#     plt.plot(values,'Orange')
#     plt.plot(full_data.Close, 'b')
#     if extra_data:
#         plt.plot(extra_dataset)
#     return fig

# st.subheader('Original Close Price and MA for 250 days')
# google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

# st.subheader('Original Close Price and MA for 200 days')
# google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

# st.subheader('Original Close Price and MA for 100 days')
# google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

# st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
# st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(x_test[['Close']])

# x_data = []
# y_data = []

# for i in range(100,len(scaled_data)):
#     x_data.append(scaled_data[i-100:i])
#     y_data.append(scaled_data[i])

# x_data, y_data = np.array(x_data), np.array(y_data)

# predictions = model.predict(x_data)

# inv_pre = scaler.inverse_transform(predictions)
# inv_y_test = scaler.inverse_transform(y_data)

# ploting_data = pd.DataFrame(
#  {
#   'original_test_data': inv_y_test.reshape(-1),
#     'predictions': inv_pre.reshape(-1)
#  } ,
#     index = google_data.index[splitting_len+100:]
# )
# st.subheader("Original values vs Predicted values")
# st.write(ploting_data)

# st.subheader('Original Close Price vs Predicted Close price')
# fig = plt.figure(figsize=(15,6))
# plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
# plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
# st.pyplot(fig)





# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App title
st.title('📈 Stock Forecast App')

# UI controls
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Cache stock data
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Unexpected data type returned from yfinance.")
        data.reset_index(inplace=True)
        required_columns = ['Date', 'Close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        st.stop()

# Load and show data
data_load_state = st.text('Loading data...')
try:
    data = load_data(selected_stock)
except Exception:
    data_load_state.text('Error loading data.')
    st.stop()
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plotting raw data
def plot_raw_data():
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
        fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting raw data: {str(e)}")
        st.stop()

plot_raw_data()

# Prepare data for Prophet with strict checks
try:
    if 'Date' in data.columns and 'Close' in data.columns:
        df_train = data[['Date', 'Close']].copy()
        df_train.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
        # Ensure 'y' is a Series
        if not isinstance(df_train['y'], pd.Series):
            raise ValueError("df_train['y'] is not a pandas Series.")
        df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
        df_train.dropna(inplace=True)

        # Validate data
        if df_train.empty:
            raise ValueError("Data is empty after cleaning.")
        if df_train['y'].isna().all():
            raise ValueError("All values in 'y' are invalid or non-numeric.")
        if len(df_train) < 2:
            raise ValueError("Not enough valid data points to train the model.")
        if not pd.api.types.is_numeric_dtype(df_train['y']):
            raise ValueError("Column 'y' contains non-numeric data after conversion.")
    else:
        raise ValueError("Dataset missing required columns: 'Date' or 'Close'.")
except Exception as e:
    st.error(f"Error preparing data for forecasting: {str(e)}")
    st.stop()

# Cache Prophet model
@st.cache_resource
def create_model():
    return Prophet()

# Train and forecast
try:
    m = create_model()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
except Exception as e:
    st.error(f"Error training or forecasting with Prophet: {str(e)}")
    st.stop()

# Show forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

# Plot forecast
st.write(f'📉 Forecast plot for {n_years} years')
try:
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
except Exception as e:
    st.error(f"Error plotting forecast: {str(e)}")
    st.stop()

# Plot forecast components
st.write("🔍 Forecast components")
try:
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
except Exception as e:
    st.error(f"Error plotting forecast components: {str(e)}")
    st.stop()
