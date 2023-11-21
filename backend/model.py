from pycaret.internal.pycaret_experiment import TimeSeriesExperiment, ClusteringExperiment
from pymongo import MongoClient
import pickle
from bson.binary import Binary
import matplotlib.pyplot as plt
import pandas as pd


class BaseModel:
    def __init__(self, data, target, session_id, db_name, collection_name):
        self.data = data
        self.target = target
        self.session_id = session_id
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_data(self, user_id):
        pickled_data = Binary(pickle.dumps(self.data))
        result=self.collection.insert_one({'user_id': user_id, 'data': pickled_data})


    def get_data(self, data_id):
        record = self.collection.find_one({'data_id': data_id})
        if record is not None:
            self.data = pickle.loads(record['data'])
        else:
            print(f"No data found for ID {data_id}.")

    def delete_data(self, data_id):
        self.collection.delete_one({'data_id': data_id})


    def load_data(self, file_path, target, data_id):
        # Determine file type
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.data = pd.read_excel(file_path)
        else:
            print("Invalid file type. Please provide a CSV or Excel file.")
            return

        # Set target
        if target in self.data.columns:
            self.target = target
        else:
            print(f"Target column {target} not found in data.")
            return

        # Save data to MongoDB
        pickled_data = Binary(pickle.dumps(self.data))
        self.collection.insert_one({'data_id': data_id, 'data': pickled_data})

    def setup(self):
        pass

    def compare_models(self):
        pass

    def save_model(self, model_type, best_model):
        pickled_model = Binary(pickle.dumps(best_model))
        self.collection.insert_one({'model_type': model_type, 'model': pickled_model})

    def load_model(self, model_type):
        record = self.collection.find_one({'model_type': model_type})
        if record is not None:
            return pickle.loads(record['model'])
        else:
            return None

class TimeSeriesModel(BaseModel):
    def setup(self):
        self.ts_exp = TimeSeriesExperiment()
        self.ts_exp.setup(data=self.data, target=self.target, session_id=self.session_id)

    def compare_models(self):
        self.model = self.ts_exp.compare_models()
        self.save_model('Time_Series')


    def plot_feature_importance(self):
        if self.model is None:
            print("No model found.")
            return None
        # Check if the model is a type that has feature importance

        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature importance
            importances = self.best_model.feature_importances_

            # Create a DataFrame for visualization
            importance_df = pd.DataFrame({
                'Feature': self.data.columns,
                'Importance': importances
            })

            # Sort by importance
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            # Plot
            importance_df.plot(kind='bar', x='Feature', y='Importance')
            plt.title('Feature Importance')
            plt.show()
        else:
            print('The best model does not support feature importance.')
            
    def forecast(self, n_periods):
        if self.model is None:
            print("No model found.")
            return None

        # Generate forecast
        forecast_df = self.ts_exp.predict_model(self.model, n_periods=n_periods)
        # Plot the last n_periods of the original data and the forecast
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index[-n_periods:], self.data[self.target][-n_periods:], label='Actual')
        plt.plot(forecast_df.index, forecast_df['Label'], label='Forecast')
        plt.legend()
        plt.show()


class ClusteringModel(BaseModel):
    def setup(self):
        self.clustering_exp = ClusteringExperiment()
        self.clustering_exp.setup(data=self.data, session_id=self.session_id)

    def compare_models(self):
        self.best_model = self.clustering_exp.compare_models()
        self.save_model('KMeans_Clustering', self.best_model)



if __name__=="main":
    # Load data
    data = get_data('sales')

    # Use the classes
    ts_model = TimeSeriesModel(data, 'Sales', 123, 'sales_forecasting', 'models')
    ts_model.setup()
    ts_model.compare_models()
    ts_model.forecast(10)


    clustering_model = ClusteringModel(data, 'Sales', 123, 'sales_forecasting', 'models')
    clustering_model.setup()
    clustering_model.compare_models()    