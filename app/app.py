from NYMEXPricePredictionApp import NYMEXPricePredictionApp
from plots import *

def main():
    app = NYMEXPricePredictionApp()

    app.display_header()
    app.display_title()
    app.display_prophet_version()
    app.display_df_ori()
    app.display_dataframe_info()
    app.display_dataframe_description()
    app.display_dataframe_head_tail()
    app.display_future_dataframe_tail()
    app.display_future_dataframe_predictions()
    
    st.header("Plots")
    c_chart(app.df, label='Close')
    plot_prophet_predictions(app.model, app.future)
    plot_prophet_components(app.model, app.future)

if __name__ == "__main__":
    main()