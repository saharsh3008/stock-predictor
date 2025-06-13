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
        # Reset index to ensure 'Date' is a column
        data = data.reset_index()
        # Flatten columns if MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[0] else col[1] for col in data.columns]
        required_columns = ['Date', 'Close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        # Debugging: Inspect raw data structure
        st.write("Debug: Raw data columns:", data.columns.tolist())
        st.write("Debug: Raw data head:", data.head())
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
    # Verify that data is a DataFrame and has required columns
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Loaded data is not a pandas DataFrame.")
    if 'Date' not in data.columns or 'Close' not in data.columns:
        raise ValueError("Dataset missing required columns: 'Date' or 'Close'.")

    # Create df_train with a copy of the required columns
    df_train = pd.DataFrame({
        'ds': data['Date'],
        'y': data['Close']
    })

    # Debugging: Inspect df_train structure
    st.write("Debug: df_train columns:", df_train.columns.tolist())
    st.write("Debug: Type of df_train['y']:", type(df_train['y']))
    st.write("Debug: df_train head:", df_train.head())

    # Ensure 'ds' is datetime
    df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
    st.write("Debug: After converting 'ds' to datetime, df_train head:", df_train.head())

    # Ensure 'y' is numeric
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
    st.write("Debug: After converting 'y' to numeric, df_train head:", df_train.head())

    # Handle missing values
    df_train.dropna(inplace=True)
    st.write("Debug: After dropping NA, df_train head:", df_train.head())

    # Validate data
    if df_train.empty:
        raise ValueError("Data is empty after cleaning.")
    if df_train['y'].isna().all():
        raise ValueError("All values in 'y' are invalid or non-numeric.")
    if len(df_train) < 2:
        raise ValueError("Not enough valid data points to train the model.")
    if not pd.api.types.is_numeric_dtype(df_train['y']):
        raise ValueError("Column 'y' contains non-numeric data after conversion.")
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
