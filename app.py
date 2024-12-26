from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

app = Flask(__name__)

# Load the CSV files and ARIMA models
monthly_data = pd.read_csv('monthly_sales_data.csv', index_col=0)
monthly_data.index = pd.to_datetime(monthly_data.index, format='%Y-%m-%d')

weekly_data = pd.read_csv('weekly_sales_data.csv', index_col=0)
weekly_data.index = pd.to_datetime(weekly_data.index, format='%Y-%m-%d')

# Load pre-trained ARIMA models for monthly and weekly data
with open('monthly_sales_model.pkl', 'rb') as f:
    monthly_model = pickle.load(f)

with open('weekly_sales_model.pkl', 'rb') as f:
    weekly_model = pickle.load(f)

@app.route('/')
def index():
    # Render the HTML template for the index page
    return render_template('index.html')

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    try:
        data = request.get_json()
        prediction_type = data.get('prediction_type')  # "monthly" or "weekly"
        periods = data.get('periods')  # Number of months or weeks to predict

        # Validate input
        if prediction_type not in ['monthly', 'weekly']:
            return jsonify({'error': 'Invalid prediction type. Use "monthly" or "weekly".'}), 400
        
        if not periods or periods <= 0:
            return jsonify({'error': 'Number of periods must be a positive integer.'}), 400

        # Choose the model based on prediction type
        if prediction_type == 'monthly':
            model = monthly_model
            forecast_data = monthly_data
            freq = 'M'
        elif prediction_type == 'weekly':
            model = weekly_model
            forecast_data = weekly_data
            freq = 'W'

        # Make the prediction
        # Use the correct offset for weekly or monthly
        if freq == 'M':
            forecast_index = pd.date_range(forecast_data.index[-1] + pd.offsets.MonthBegin(),
                                          periods=periods, freq=freq)
        elif freq == 'W':
            forecast_index = pd.date_range(forecast_data.index[-1] + pd.DateOffset(weeks=1),
                                          periods=periods, freq=freq)
        
        forecast_values = model.get_forecast(steps=periods).predicted_mean
        forecast_values = np.maximum(forecast_values, 0)
        confidence_intervals = model.get_forecast(steps=periods).conf_int()

        # Create a plot for the forecasted values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast_data.index, forecast_data['Count'], label='Historical Sales', color='blue')
        ax.plot(forecast_index, forecast_values, label='Forecasted Sales', color='red')
        ax.fill_between(forecast_index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='gray', alpha=0.3)
        ax.set_title(f'{prediction_type.capitalize()} Sales Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()

        # Save the plot to a BytesIO object and convert it to base64
        img_io = BytesIO()
        fig.savefig(img_io, format='png')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # Return the results as JSON with the graph
        forecast_results = {
            'forecast_index': forecast_index.strftime('%Y-%m-%d').tolist(),
            'forecast_values': forecast_values.tolist(),
            'confidence_intervals': confidence_intervals.values.tolist(),
            'forecast_plot': img_base64  # Include the plot as a base64-encoded string
        }

        return jsonify(forecast_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
