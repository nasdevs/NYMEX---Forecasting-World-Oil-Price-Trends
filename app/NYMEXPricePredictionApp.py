import io
from importlib_metadata import version
import pandas as pd
import prophet
import streamlit as st

class NYMEXPricePredictionApp:
    def __init__(self, data_path='../data/brentcrudeoil-dailybrentoil.csv'):
        self.df = self._load_and_preprocess_data(data_path)
        self.df_ori = self.df.copy()

        self.model = self._initialize_prophet_model()
        self.future = self._create_future_dataframe()

    def _load_and_preprocess_data(self, data_path):
        df = pd.read_csv(data_path)
        df['Open'] = df.shift(1).Close
        df = df.reindex(columns=['Date', 'Open', 'Close', 'chg(close)', 'Low', 'chg(low)', 'High', 'chg(high)'])
        df.dropna(inplace=True)
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        return df

    def _initialize_prophet_model(self):
        model = prophet.Prophet()
        model.fit(self.df)
        return model

    def _create_future_dataframe(self, periods=365):
        return self.model.make_future_dataframe(periods=periods)

    def display_header(self):
        st.header(self.df.columns)

    def display_title(self):
        st.title("NYMEX Price Prediction")

    def display_prophet_version(self):
        st.subheader("Prophet Version")
        st.markdown(f'Currently used `prophet` library version is `{version("prophet")}`')
        st.markdown('''---''')

    def display_df_ori(self):
        st.subheader("Original Data")
        st.table(self.df_ori.head())