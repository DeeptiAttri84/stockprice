# AAPL Stock Prediction System

A comprehensive stock price prediction system using machine learning models (LSTM, Decision Tree, Random Forest) with an interactive web dashboard built using Dash and Plotly.

##  Features

### Interactive Dashboard (app1.py)
- **Real-time Predictions**: Interactive slider to select dates and get LSTM predictions
- **Visual Analytics**: Dynamic charts showing actual vs predicted prices
- **Profit/Loss Calculation**: Automatic calculation of expected profit or loss
- **Authentication**: Basic authentication system for secure access
- **Responsive UI**: Clean, user-friendly interface with Plotly graphs

### Model Comparison System (main.py)
- **Multiple ML Models**: LSTM, Decision Tree, and Random Forest comparison
- **Data Preprocessing**: Advanced preprocessing including outlier removal and feature engineering
- **Performance Metrics**: Comprehensive evaluation with MSE, MAE, R², Precision, Recall, F1-score
- **Visualization**: Side-by-side comparison plots for all models

##  Requirements

```
dash
plotly
pandas
numpy
scikit-learn
tensorflow
keras
matplotlib
dash-auth
```

##  Installation

1. Clone the repository or download the files
2. Install required packages:
```bash
pip install dash plotly pandas numpy scikit-learn tensorflow matplotlib dash-auth
```
3. Ensure you have the AAPL.csv dataset in the correct path

## Dataset Structure

Your AAPL.csv should contain the following columns:
- `date`: Date column (will be converted to datetime)
- `open`: Opening price
- `high`: Highest price of the day
- `low`: Lowest price of the day
- `close`: Closing price
- `volume`: Trading volume
- `SYMBOL`: Stock symbol (optional, will be dropped)

## Usage

### Running the Interactive Dashboard

```bash
python app1.py
```

**Features:**
- Navigate to `http://127.0.0.1:8050/` in your browser
- Login with username: `deepti`, password: `123`
- Use the date slider to select prediction dates
- View real-time predictions and profit/loss calculations
- Analyze interactive charts showing actual vs predicted prices

### Running the Model Comparison

```bash
python main.py
```

**What it does:**
- Loads and preprocesses the data with advanced techniques
- Removes outliers using IQR method
- Applies PowerTransformer for skewness correction
- Trains three different models (LSTM, Decision Tree, Random Forest)
- Generates comprehensive performance metrics
- Creates comparison visualizations

##  Model Architecture

### LSTM Model (Dashboard)
- **Architecture**: 2 LSTM layers (50 units each) with Dropout (0.2)
- **Lookback Period**: 60 days
- **Features**: Open, High, Low, Volume, Close prices
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

### LSTM Model (Comparison)
- **Architecture**: 3 LSTM layers (256→128→64 units) with Dropout (0.4)
- **Features**: Open, High, Low, Close, MA_50, MA_200, Volume
- **Optimizer**: Nadam
- **Training**: 100 epochs with batch size 64

### Traditional ML Models
- **Decision Tree**: Max depth 5, prevents overfitting
- **Random Forest**: Ensemble method with max depth 5

## Performance Metrics

The system evaluates models using:
- **MSE (Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **R² Score**: Coefficient of determination
- **Precision/Recall/F1-Score**: Classification-based metrics for directional accuracy

##  Advanced Features

### Data Preprocessing
- **Outlier Detection**: IQR-based outlier removal
- **Feature Engineering**: Moving averages (50-day and 200-day)
- **Data Transformation**: PowerTransformer for skewness correction
- **Scaling**: StandardScaler and MinMaxScaler normalization
- **Correlation Analysis**: Automatic detection of highly correlated features

### Technical Indicators
- **MA_50**: 50-day moving average
- **MA_200**: 200-day moving average
- **Volume Analysis**: Trading volume as a feature

##  File Structure

```
project/
├── app1.py              # Interactive Dash dashboard
├── main.py          # Model comparison script
├── AAPL.csv             # Stock data (required)
└── README.md            # This file
```

## Security

- Basic authentication implemented in dashboard
- Credentials: Username: `deepti`, Password: `123`
- **Note**: Change default credentials for production use

## Key Insights

### Dashboard Capabilities
- **Interactive Predictions**: Real-time LSTM predictions based on user-selected dates
- **Profit/Loss Analysis**: Automatic calculation comparing predicted vs previous closing price
- **Visual Analytics**: Dynamic plotting with actual and predicted price lines
- **Data Storage**: Efficient data handling using Dash Store components

### Model Performance Analysis
- **Comprehensive Comparison**: Three different ML approaches
- **Advanced Preprocessing**: Handles real-world data challenges
- **Multiple Metrics**: Holistic performance evaluation
- **Visual Comparison**: Side-by-side prediction plots

## Important Notes

1. **Data Path**: Update the CSV file path in main.py to match your system
2. **Memory Requirements**: Deep LSTM models require sufficient RAM
3. **Training Time**: Model comparison script may take several minutes to complete
4. **Authentication**: Change default credentials before deployment

## 🔮 Future Enhancements

- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Implement ensemble methods combining multiple models
- Add real-time data fetching from financial APIs
- Include sentiment analysis from news data
- Implement advanced time series validation techniques

