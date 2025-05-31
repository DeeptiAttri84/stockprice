import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, precision_score, recall_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# ---- Load and Preprocess Data ----
# Load the dataset
df = pd.read_csv(r'C:\Users\Deepti\Desktop\programs\AAPL.csv')
print(df.info())
print(df.head())

# Drop non-numeric and irrelevant columns, if necessary
df = df.drop(columns=['SYMBOL'], errors='ignore')

# Drop columns with all NaN values
df = df.dropna(axis=1, how='all')

# Select numeric columns only
numeric_df = df.select_dtypes(include='number')

# Calculate Q1, Q3, and IQR
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

# Define the outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_clean = df[~((numeric_df < lower_bound) | (numeric_df > upper_bound)).any(axis=1)]
print(df_clean.head())

# Step 1: Calculate the correlation matrix
correlation = numeric_df.corr()

# Step 2: Use absolute values for easier analysis
correlation = correlation.abs()

# Step 3: Flatten the correlation matrix to analyze pairs
correlation_flat = (
    correlation.unstack()
    .reset_index()
    .rename(columns={"level_0": "Column1", "level_1": "Column2", 0: "Correlation"})
)

# Step 4: Filter out self-correlations
correlation_flat = correlation_flat[correlation_flat['Column1'] != correlation_flat['Column2']]

# Step 5: Sort by the highest correlation
high_corr_pairs = correlation_flat.sort_values(by="Correlation", ascending=False)

# Step 6: Display pairs with a high correlation (e.g., above 0.9)
print(high_corr_pairs[high_corr_pairs['Correlation'] > 0.9])

# Check required columns
if 'date' not in df.columns or 'close' not in df.columns:
    raise KeyError("'date' or 'close' column is missing from the dataset. Please check the column names.")

# Ensure 'date' column is in datetime format without timezone
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
df = df.dropna(subset=['date'])  # Drop rows where 'date' couldn't be converted

# Fill missing numeric values with column means
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Calculate moving averages
df['MA_50'] = df['close'].rolling(window=50).mean()
df['MA_200'] = df['close'].rolling(window=200).mean()

# Drop rows with NaN values resulting from moving averages
df = df.dropna(subset=['MA_50', 'MA_200'])

skewness_before = numeric_df.skew()
print("\nSkewness before transformation:\n", skewness_before)

# Apply PowerTransformer
power_transformer = PowerTransformer()
transformed_features = power_transformer.fit_transform(numeric_df)
numeric_df = pd.DataFrame(transformed_features, columns=numeric_df.columns)

# Skewness after transformation
skewness_after = numeric_df.skew()
print("\nSkewness after transformation:\n", skewness_after)

# Replace original numeric columns with transformed values
df[numeric_df.columns] = numeric_df

# Select relevant columns
df = df[['date', 'open', 'high', 'low', 'close', 'MA_50', 'MA_200', 'volume']]

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'MA_50', 'MA_200', 'volume']])
df[['open', 'high', 'low', 'close', 'MA_50', 'MA_200', 'volume']] = scaled_features
print(df.info())

# Preserve 'date' and 'close' for evaluation and plotting
df_with_dates = df[['date', 'close']].copy()

# Split features and target variable
X = df.drop(['date', 'close'], axis=1)
y = df['close']


# Split data into training and testing sets (time-series split without shuffling)
X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
    X,y,df_with_dates['date'], test_size=0.2, shuffle=False
)

# Ensure target variables are in float32 format
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# ---- LSTM Model ----
def train_lstm(X_train, y_train, X_test, y_test):
    # Reshape features for LSTM model
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.4),
        LSTM(128, return_sequences=True),
        Dropout(0.4),
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(1, activation='linear')
    ])
    model.summary()
    model.compile(optimizer='Nadam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train_lstm, y_train, epochs=100, batch_size=64, validation_data=(X_test_lstm, y_test))
    return model.predict(X_test_lstm).reshape(-1)

# Train and evaluate LSTM
y_pred_lstm = train_lstm(X_train, y_train, X_test, y_test)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
accuracy_lstm = 1 - (mse_lstm / np.var(y_test))

threshold = 0.5 * y_test
y_pred_binary_lstm = np.abs(y_pred_lstm - y_test) <= threshold
y_test_binary = np.abs(y_test - y_test) <= threshold

precision_lstm = precision_score(y_test_binary, y_pred_binary_lstm)
recall_lstm = recall_score(y_test_binary, y_pred_binary_lstm)
f1_lstm = f1_score(y_test_binary, y_pred_binary_lstm)

print("\nLSTM Model:")
print("MSE:", mse_lstm)
print("MAE:", mae_lstm)
print("Accuracy:", accuracy_lstm)
print("Precision:", precision_lstm)
print("Recall:", recall_lstm)
print("F1-score:", f1_lstm)

model_dt = DecisionTreeRegressor(random_state=42,max_depth=5)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
accuracy_dt = model_dt.score(X_test, y_test)
threshold = 0.5 * y_test
y_pred_binary_dt = np.abs(y_pred_dt - y_test) <= threshold
y_test_binary = np.abs(y_test - y_test) <= threshold

precision_dt = precision_score(y_test_binary, y_pred_binary_dt)
recall_dt = recall_score(y_test_binary, y_pred_binary_dt)
f1_dt = f1_score(y_test_binary, y_pred_binary_dt)

print("\nDecision Tree Model:")
print("MSE:", mse_dt)
print("MAE:", mae_dt)
print("R²:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1-score:", f1_dt)


# ---- Random Forest Model ----
model_rf = RandomForestRegressor(random_state=42,max_depth=5)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
threshold = 0.5 * y_test
y_pred_binary_rf = np.abs(y_pred_rf - y_test) <= threshold
y_test_binary = np.abs(y_test - y_test) <= threshold

precision_rf = precision_score(y_test_binary, y_pred_binary_rf)
recall_rf = recall_score(y_test_binary, y_pred_binary_rf)
f1_rf = f1_score(y_test_binary, y_pred_binary_rf)

print("\nRandom Forest Model:")
print("MSE:", mse_rf)
print("MAE:", mae_rf)
print("R²:", r2_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-score:", f1_rf)

# ---- Plot Results ----
# LSTM Predictions
plt.figure(figsize=(14, 7))
plt.plot(date_test, y_test, label='Actual Prices', color='blue')
plt.plot(date_test, y_pred_lstm, label='Predicted Prices (LSTM)', color='purple')
plt.title('LSTM Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Decision Tree Predictions
plt.figure(figsize=(14, 7))
plt.plot(date_test, y_test, label='Actual Prices', color='blue')
plt.plot(date_test, y_pred_dt, label='Predicted Prices (DT)', color='green')
plt.title('Decision Tree Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Random Forest Predictions
plt.figure(figsize=(14, 7))
plt.plot(date_test, y_test, label='Actual Prices', color='blue')
plt.plot(date_test, y_pred_rf, label='Predicted Prices (RF)', color='orange')
plt.title('Random Forest Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




