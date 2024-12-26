# TV Sales Forecasting

This repository contains the code and documentation for the **TV Sales Forecasting** project. The project is designed to analyze and predict weekly and monthly TV sales using time-series models like ARIMA, with data provided by a leading Bangladeshi brand.

---

## **Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/nomanvb/tv-sales-forecast)
- **Period**: January 2014 - August 2016
- **Columns**:
  - `Date`: Date of sale
  - `Model`: TV model identifier (anonymized)
  - `Count`: Quantity sold

---

## **Objective**
The project aims to:
- Predict weekly and monthly TV sales.
- Identify sales trends and seasonal patterns.
- Support operational planning, inventory management, and strategic decision-making.

---

## **Features**
- **Time-Series Analysis**: Sales trends across months, weeks, and days.
- **Forecasting Models**: 
  - Monthly forecasts using ARIMA (3, 1, 1).
  - Weekly forecasts using ARIMA (2, 1, 2).
- **Visualization**: Intuitive plots for sales trends and forecasts with confidence intervals.
- **Lagged Features**: Incorporating historical data for robust forecasting.

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/tv-sales-forecasting.git

## **Usage**
1. Preprocess the sales data:
  - Extract time-based features: Year, Month, Day, etc.
  - Generate lagged and rolling features.
2. Fit ARIMA models for both weekly and monthly data.
3. Visualize and analyze forecast results:
     ```bash
     python app.py
4. Access the forecast API.

## **Results**
- Clear insights into sales trends during significant periods (e.g., EID festivals).
- Improved planning through accurate sales forecasting.

