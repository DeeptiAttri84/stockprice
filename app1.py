import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import dash_auth

# ---- Load and Preprocess Data ----
# Load the dataset
df = pd.read_csv('AAPL.csv')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Features for prediction (using 'close')
features = ['open', 'high', 'low', 'volume', 'close'] 

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(df[features])

# Create the dataset with look back (e.g., 60 days)
look_back = 60
X, Y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i])
    Y.append(scaled_data[i, 0])  # Assuming 'open' is the first feature in 'features' list
X, Y = np.array(X), np.array(Y)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], len(features)))

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ---- LSTM Model ----
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], len(features))))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1)) 

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, Y_train, epochs=10, batch_size=32)

# ---- Dash App ----
app = dash.Dash(__name__)

# Authentication (replace with your actual usernames and passwords)
VALID_USERNAME_PASSWORD_PAIRS = {
    'deepti': '123'
}
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

app.layout = html.Div([
    html.H1("AAPL Stock Prediction Dashboard"),
     html.H4("Date slider"),
    dcc.Slider(
        id='date-slider',
        min=0,
        max=len(df) - 1,
        step=1,  # Changed to Slider
        value=len(df) - 1,  # Initial value is the last date
        marks={i: str(date) for i, date in enumerate(df['date'])} 
    ),
    html.Div(id='predicted-price'),
    dcc.Graph(id='lstm-graph'),

    # Store the DataFrame in a dcc.Store component
    dcc.Store(id='data-store', data=df.to_dict('records')) 
])


@app.callback(
    [Output('predicted-price', 'children'),
     Output('lstm-graph', 'figure')],
    [Input('date-slider', 'value'),  # Changed to Slider input
     State('data-store', 'data')]
)
def update_prediction(selected_date_index, data): 
    try:
        df = pd.DataFrame(data)

        # Get the selected date from the slider index
        selected_date = df['date'].iloc[selected_date_index] 

        # Filter data up to the selected date
        filtered_df = df[df['date'] <= selected_date].copy()

        # Ensure we have enough data for prediction (lookback period)
        if len(filtered_df) < look_back:
            return "Not enough data for prediction. Please select a later date.", go.Figure()

        # Prepare data for LSTM prediction (using the last 'look_back' days)
        inputs = filtered_df[features].iloc[-look_back:].values
        inputs = scaler.transform(inputs)  # Scale the inputs

        # Reshape the input for LSTM
        X_test = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))  # Shape should be (1, look_back, num_features)

        # Generate prediction for the selected date
        predicted_price = lstm_model.predict(X_test)

        # Inverse transform to get the actual price
        predicted_price = scaler.inverse_transform(
            np.concatenate((predicted_price, np.zeros((1, len(features)-1))), axis=1)
        )[0][0]  # Get the first value (open price)

        # Get the previous day's closing price (last row in filtered_df)
        previous_close_price = filtered_df['close'].iloc[-1]

        # Determine if it's a profit or loss
        if predicted_price > previous_close_price:
            profit_or_loss = f"Profit: ₹{predicted_price - previous_close_price:.2f}"
        else:
            profit_or_loss = f"Loss: ₹{previous_close_price - predicted_price:.2f}"

        # Create the LSTM graph
        fig_lstm = go.Figure()
        fig_lstm.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['open'], mode='lines', name='Actual Prices'))

        # Add the predicted line (extend the actual prices with the prediction)
        future_date = pd.to_datetime(selected_date) 
        future_df = pd.DataFrame({'date': [future_date], 'open': [predicted_price]})
        predicted_line_df = pd.concat([filtered_df, future_df], ignore_index=True)

        fig_lstm.add_trace(go.Scatter(x=predicted_line_df['date'], y=predicted_line_df['open'], mode='lines', name='Predicted Line', line=dict(dash='dot')))

        # Add a marker for the predicted point
        fig_lstm.add_trace(go.Scatter(x=[selected_date], y=[predicted_price], mode='markers', name='Predicted Point', marker=dict(size=10, color='red')))

        fig_lstm.update_layout(title='LSTM Predictions', xaxis_title='Date', yaxis_title='Price')

        return f"Predicted price for {selected_date}: ₹{predicted_price:.2f}. {profit_or_loss}", fig_lstm

    except Exception as e:
        print(f"Error in update_prediction: {e}")
        return "Error occurred", go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)