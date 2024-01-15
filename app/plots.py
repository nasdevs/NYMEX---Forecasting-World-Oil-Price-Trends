import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plot_prophet_predictions(model, future):
    st.subheader("Prophet Predictions")
    fig = model.plot(model.predict(future))
    st.pyplot(fig)

def plot_prophet_components(model, future):
    st.subheader("Prophet Components")
    fig = model.plot_components(model.predict(future))
    st.pyplot(fig)

def c_chart(data, label):
    candlestick = go.Figure(data=[go.Candlestick(x=data.index,
                                                  open=data['Open'],
                                                  high=data['High'],
                                                  low=data['Low'],
                                                  close=data['y'])])
    candlestick.update_xaxes(title_text='Time', rangeslider_visible=True)

    candlestick.update_layout(
        title={
            'text': '{:} Candlestick Chart'.format(label),
            'y': 0.8,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    candlestick.update_yaxes(title_text='Price in USD', ticksuffix='$')
    st.subheader("Candlestick Chart")
    st.plotly_chart(candlestick)
