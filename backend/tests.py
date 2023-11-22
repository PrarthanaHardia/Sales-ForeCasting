import pandas as pd
from models import BaseModel, TimeSeriesModel, ClusteringModel

def test_base_model():
    # Initialize BaseModel
    base_model = BaseModel(123, 'test_db', 'test_collection')

    # Load data
    base_model.load_data('test.csv', 'target', 'test_data')

    # Get data
    base_model.get_data('test_data')
    assert isinstance(base_model.data, pd.DataFrame), "Data is not loaded correctly."

    # Delete data
    base_model.delete_data('test_data')
    base_model.get_data('test_data')
    assert base_model.data is None, "Data is not deleted correctly."

def test_time_series_model():
    # Initialize TimeSeriesModel
    ts_model = TimeSeriesModel(123, 'test_db', 'test_collection')

    # Load data
    ts_model.load_data('test.csv', 'target', 'test_data')

    # Setup
    ts_model.setup()

    # Compare models
    ts_model.compare_models()

    # Forecast
    forecast_df = ts_model.forecast(10)
    assert isinstance(forecast_df, pd.DataFrame), "Forecast is not generated correctly."

def test_clustering_model():
    # Initialize ClusteringModel
    clustering_model = ClusteringModel(123, 'test_db', 'test_collection')

    # Load data
    clustering_model.load_data('test.csv', 'target', 'test_data')

    # Setup
    clustering_model.setup()

    # Compare models
    clustering_model.compare_models()

if __name__ == "__main__":
    test_base_model()
    test_time_series_model()
    test_clustering_model()