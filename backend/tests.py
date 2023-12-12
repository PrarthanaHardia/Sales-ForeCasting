import pandas as pd
from pmdarima import auto_arima
from pmdarima.arima.utils import ndiffs
from scipy import stats

def sales_forecasting(data, target_column):
    # Find and set timestamp
    timestamp_columns = data.select_dtypes(include=['datetime64']).columns
    if timestamp_columns.empty:
        return {"error": "No timestamp found"}

    primary_timestamp_column = timestamp_columns[0]
    data.set_index(primary_timestamp_column, inplace=True)

    # Handle missing values
    data.fillna(method="ffill", inplace=True)

    # Identify and remove rows with missing or problematic values
    data.dropna(inplace=True)

    # Outlier detection and removal using Z-score
    z_scores = stats.zscore(data.select_dtypes(include=['float64', 'int64']))
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]

    # Check and handle stationarity
    for column in data.columns:
        d = ndiffs(data[column], test='adf')
        if d > 0:
            data[column] = data[column].diff(d)

    # Train auto ARIMA model
    try:
        model = auto_arima(data[target_column], suppress_warnings=True)
    except:
        return {"error": "Error in training auto ARIMA model"}

    return {"status": "success", "model": model}

# Example usage
# Assuming 'data' is your input DataFrame and 'sales' is the target column
data = pd.read_csv("your_data.csv")  # Replace with your data loading logic
result = sales_forecasting(data, target_column='sales')
print(result)
import pandas as pd
from pmdarima import auto_arima
from pmdarima.arima.utils import ndiffs
import matplotlib.pyplot as plt
import io
import base64
import pymongo
from pymongo import MongoClient

def save_model_to_mongodb(model, database_name, collection_name):
    client = MongoClient('your_mongodb_connection_string')
    db = client[database_name]
    collection = db[collection_name]

    model_data = io.BytesIO()
    model.to_pkl(model_data)
    model_data.seek(0)

    model_dict = {
        "model_type": "auto_arima",
        "model_data": model_data.read(),
    }

    collection.insert_one(model_dict)
    client.close()

def load_model_from_mongodb(database_name, collection_name):
    client = MongoClient('your_mongodb_connection_string')
    db = client[database_name]
    collection = db[collection_name]

    model_dict = collection.find_one({"model_type": "auto_arima"})

    if model_dict:
        model_data = io.BytesIO(model_dict["model_data"])
        model = auto_arima.from_pkl(model_data)
        client.close()
        return model
    else:
        client.close()
        return None

def generate_forecast_graph(model, data, target_column, forecast_horizon):
    # Handle missing values
    data.fillna(method="ffill", inplace=True)

    # Identify and remove rows with missing or problematic values
    data.dropna(inplace=True)

    # Check and handle stationarity
    for column in data.columns:
        d = ndiffs(data[column], test='adf')
        if d > 0:
            data[column] = data[column].diff(d)

    # Forecast using the loaded model
    forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)

    # Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[target_column], label='Actual Data')
    plt.plot(pd.date_range(start=data.index[-1], periods=forecast_horizon+1, freq=data.index.freq)[1:],
             forecast, label='Forecast', color='red')
    plt.fill_between(pd.date_range(start=data.index[-1], periods=forecast_horizon+1, freq=data.index.freq)[1:],
                     conf_int[:, 0], conf_int[:, 1], color='red', alpha=0.2, label='Confidence Interval')
    plt.title('Sales Forecast')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

# Example usage
# Assuming 'data' is your input DataFrame and 'sales' is the target column
data = pd.read_csv("your_data.csv")  # Replace with your data loading logic
model = sales_forecasting(data, target_column='sales')['model']
save_model_to_mongodb(model, database_name='your_db_name', collection_name='your_collection_name')

loaded_model = load_model_from_mongodb(database_name='your_db_name', collection_name='your_collection_name')

if loaded_model:
    generate_forecast_graph(loaded_model, data, target_column='sales', forecast_horizon=10)
else:
    print("Model not found in MongoDB.")
