from flask import Flask, request, jsonify, url_for
from flask_restful import Resource, Api
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
from pycaret.clustering import ClusteringExperiment
import dill
from bson.binary import Binary
from werkzeug.utils import secure_filename
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import uuid
from flask import Flask, jsonify
from flask_restful import Api, Resource
from statsmodels.tsa.seasonal import seasonal_decompose
from bson import Binary,ObjectId
import io
import base64
import urllib
from pycaret.time_series import load_experiment as ts_load_experiment
from pmdarima import auto_arima
from pmdarima.arima.utils import ndiffs
from scipy import stats
import numpy as np


app = Flask((__name__))
CORS(app)
api=Api(app) 

client = MongoClient("mongodb://localhost:27017")  
db = client["Prospectify"]  
dataset_collection = db["dataset"]
model_collection = db["model"]
user_data=db["user_data"]
app.config['dataset_collection'] = db["dataset"]
app.config['model_collection'] = db["model"]
app.config['user_data'] = db["user_data"]

ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class check(Resource):
    def get(self):
        return {"status":True,"message":"Server is running"}


def check_validity(user_id, dataset_id):
    user_record = user_data.find_one({"user_id": user_id})
    if user_record:
        dataset_record = dataset_collection.find_one({"_id": ObjectId(dataset_id)})

        if dataset_record:
            print(dataset_record["content_type"])
            if dataset_record["content_type"] == "text/csv":
                df = pd.read_csv(io.BytesIO(dataset_record["file_data"]))
            elif dataset_record["content_type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(io.BytesIO(dataset_record["file_data"]))
            else:
                return {"status": False, "message": "Invalid file format"}
            return df
        else:
            return {"status": False, "message": "Invalid dataset_id"}
    return {"status": False, "message": "Invalid user_id"}

class file_upload(Resource):
    def post(self):
        try:
            user_id = request.form['user_id']
            uploaded_file = request.files['file']
            file_data = uploaded_file.read()
            if uploaded_file and allowed_file(uploaded_file.filename):
                if len(file_data) > MAX_CONTENT_LENGTH:
                    return jsonify({
                        "status": False,
                        "message": "File size exceeds the maximum allowed size (16MB)"
                    })

                filename = secure_filename(uploaded_file.filename)
                content_type = uploaded_file.content_type
                uploaded_file.seek(0)                
                time_stamp=datetime.now()
                result = dataset_collection.insert_one({"file_data": file_data,"time_stamp":time_stamp, "filename": filename, "content_type": content_type, "user_id": user_id})
                file_id = str(result.inserted_id)


                return jsonify({
                    "status": True,
                    "message": "File uploaded",
                    "file_id": file_id,
                })
            else:
                return jsonify({
                    "status": False,
                    "message": f"Invalid file format. Allowed formats:{' '.join(ALLOWED_EXTENSIONS)}"
                })
        except Exception as e:
            return jsonify({
                "status": False,
                "message": str(e)
            })

class Login(Resource):
    def post(self):
        user_data = request.get_json()
        user_id = user_data.get('email')
        password = user_data.get('password')

        # Check if user_id and password are valid
        # Save user data to MongoDB
        if validate_credentials(email, password):
            # self.save_user_data(user_id, password)
            # Return success response
            return {'message': 'Login successful'}, 200
        else:
            return {'message': 'Invalid credentials'}, 401

def validate_credentials(email, password):
    # Retrieve the user data from the database based on the user_id
    user = user_data.find_one({'user_id': email})

    # Check if the user exists and the password matches
    if user and user['password'] == password:
        return True
    else:
        return False

class Signup(Resource):
    def post(self):
        signup_data = request.get_json()
        name = signup_data.get('name')
        email = signup_data.get('email')
        password = signup_data.get('password')
        print(signup_data)

        # Check if email already exists in the database
        if check_email_exists(email):
            return {'message': 'Email already exists'}, 400

        # Save user data to MongoDB

        user_id=save_user_data(name, email,password)

        # Return success response
        return {'message': 'Signup successful',"user_id":user_id}, 200

def check_email_exists( email):
    # Check if email exists in the database
    existing_user = user_data.find_one({'email': email})
    return existing_user is not None

def save_user_data(name, email,password):
    # Save user data to MongoDB
    user_id = str(uuid.uuid4())
    signup_data = {'user_id': user_id, 'email': email, 'password': password, 'name': name}
    user_data.insert_one(signup_data)

api.add_resource(file_upload,"/upload")
api.add_resource(Login, "/login")
api.add_resource(Signup, "/signup")

class BaseModel:
    def __init__(self, data, target, session_id, db_name, collection_name):
        self.data = data
        self.target = target
        self.session_id = session_id


    def save_data(self, user_id):
        dilld_data = Binary(dill.dumps(self.data))
        result=self.collection.insert_one({'user_id': user_id, 'data': dilld_data})


    def save_experiment_to_binary(self, experiment):
        # Save the experiment to a BytesIO object
        experiment_binary_io = io.BytesIO()
        experiment.save_experiment(experiment_binary_io)
        experiment_binary_io.seek(0)
        
        # Convert the BytesIO object to binary data
        experiment_binary = Binary(experiment_binary_io.read())
        experiment_binary_io.close()

        return experiment_binary


    def get_data(self, data_id):
        record = self.collection.find_one({'data_id': data_id})
        if record is not None:
            self.data = dill.loads(record['data'])
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
        dilld_data = Binary(dill.dumps(self.data))
        self.collection.insert_one({'data_id': data_id, 'data': dilld_data})

    def setup(self):
        pass

    def compare_models(self):
        pass

    def save_model(self, model_type, model, experiment, dataset_id):
        self.model=Binary(dill.dumps(model))
        model_record = model_collection.find_one({'model_type': model_type, 'dataset_id': dataset_id})
        if model_record is not None:
            print("here")
            model_collection.update_one(
                {'model_type': model_type, 'dataset_id': dataset_id},
                {'$set': {'model_data': experiment, 'model': self.model}}
            )
        else:
            model_collection.insert_one({'model_type': model_type, 'model_data': experiment, 'model': self.model, 'dataset_id': dataset_id, 'target': self.target})

    def load_model(self, model_type, dataset_id):
        record = self.collection.find_one({'model_type': model_type, 'dataset_id': dataset_id})
        if record is not None:
            return dill.loads(record['model'])
        else:
            return None

class TimeSeriesModel(BaseModel):
    def setup(self):
        self.ts_exp = TSForecastingExperiment()
        # Identify columns that might contain dates and convert them to datetime
        object_columns = self.data.select_dtypes(include=['object']).columns

        # Identify columns that might contain dates and convert them to datetime
        for column in object_columns:
            try:
                self.data[column] = pd.to_datetime(self.data[column])
            except (TypeError, ValueError):
                pass  # Ignore columns that cannot be converted timestamp
            
        timestamp_columns = self.data.select_dtypes(include=['datetime64']).columns
        if timestamp_columns.any():
            primary_timestamp_column = timestamp_columns[0]
            self.data.drop(columns=list(timestamp_columns[1:]), inplace=True)
            self.data = self.data.resample('D', on=timestamp_columns[0]).sum().reset_index()
            self.data.set_index(primary_timestamp_column, inplace=True)

        self.data.fillna(method="ffill", inplace=True)
        self.ts_exp.setup(data=self.data, target=self.target, session_id=self.session_id,numeric_imputation_target='mean', numeric_imputation_exogenous=True)

    def compare_models(self,data_id):
        # self.model = self.ts_exp.compare_models()
        self.model=self.ts_exp.create_model('auto_arima')
        exp=self.save_experiment_to_binary(self.ts_exp)
        self.save_model('Time_Series',self.model, exp, data_id)

    def plot_feature_importance(self):
        if self.model is None:
            print("No model found.")
            return None
        # Check if the model is a type that has feature importance

        if hasattr(self.model, 'feature_importances_'):
            # Get feature importance
            importances = self.model.feature_importances_

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

    def calculate_summary_stats(self):
        if self.model is None:
            print("No model found.")
            return None

        # Generate forecast
        forecast_df = self.ts_exp.predict_model(self.model,fh=24)
        # Calculate summary statistics
        summary_stats = forecast_df.describe()

        return summary_stats.to_dict()

    def calculate_seasonal_distribution(self):
        if self.model is None:
            print("No model found.")
            return None

        # Generate forecast
        forecast_df = self.ts_exp.predict_model(self.model, n_periods=len(self.data))
        # Calculate seasonal distribution
        seasonal_distribution = forecast_df.groupby(forecast_df.index.month).mean()

        return seasonal_distribution


class ClusteringModel(BaseModel):
    def setup(self):
        self.clustering_exp = ClusteringExperiment()
        self.clustering_exp.setup(data=self.data, session_id=self.session_id)

    def compare_models(self):
        self.best_model = self.clustering_exp.create_model("kmeans")
        self.save_model('KMeans_Clustering', self.best_model)




class FPsummary(Resource):
    def post(self):
        # Retrieve the forecast plot summary for the given data_id
        data=request.get_json()
        data_id = data['data_id']
        n=int(data['n_periods'])
        user_id=data['user_id']
        target=data['target']

        validity=check_validity(user_id,data_id)
        if isinstance(validity,pd.DataFrame):
            dataset=validity
        else:
            return validity

        # Retrieve the associated model
        model = model_collection.find_one({'dataset_id': data_id,'model_type':"Time_Series"},{'model_data':1,"model":1})
        if model is None:
            return {'error': 'Model not found'}, 404

        # # Load the experiment
        exp_model= dill.loads(model["model"])
        data=dataset
        object_columns = data.select_dtypes(include=['object']).columns

        # Identify columns that might contain dates and convert them to datetime
        for column in object_columns:
            try:
                data[column] = pd.to_datetime(data[column])
            except (TypeError, ValueError):
                pass  # Ignore columns that cannot be converted timestamp
            
        timestamp_columns = data.select_dtypes(include=['datetime64']).columns
        if timestamp_columns.any():
            primary_timestamp_column = timestamp_columns[0]
            data.drop(columns=list(timestamp_columns[1:]), inplace=True)
            data.set_index(primary_timestamp_column, inplace=True)
            data = data.resample('D').sum()
            print(data.head())
            data.index=pd.to_datetime(data.index)
            # data.set_index(primary_timestamp_column, inplace=True)


        data.fillna(method="ffill", inplace=True)
        data.index=data.index.to_period("D")

        exp = ts_load_experiment(io.BytesIO(model["model_data"]),data=data)

        actual_data = {str(index): value for index, value in zip(data.index[-n:], data[target][-n:])} 

        forecast_data = pd.DataFrame(index=range(len(data), len(data) + n))

        # Use predict_model to make forecasts
        print(data.tail(n))
        print(exp.predict_model(estimator=exp_model, fh=n,X=data.tail(n)))
        forecast_values = exp.predict_model(estimator=exp_model, fh=n)
        
        summary_stats = forecast_values.describe().to_dict()

        # Prepare forecast data for the frontend
        forecast_data = {str(index): value for index, value in zip(range(len(data), len(data) + n), forecast_values['Label'])}

        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(data), len(data) + n), forecast_values['Label'])
        plt.title('Forecast')
        plt.xlabel('Time')
        plt.ylabel('Forecast')

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert the BytesIO object to a base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Create a data URL from the base64 string
        image_data_url = 'data:image/png;base64,' + urllib.parse.quote(image_base64)

        return {"status": True, "summary_stats":summary_stats, "actual_data": actual_data, "forecast_data": forecast_data, "forecast_plot": image_data_url}
class SeasonalDistribution(Resource):
    def get(self):
        # Retrieve the dataset with the given data_id
        data = request.get_json()
        data_id = data['data_id']
        user_id = data['user_id']
        dataset = dataset_collection.find_one({'_id': ObjectId(data_id)})
        validity = check_validity(user_id, data_id)
        if isinstance(validity, pd.DataFrame):
            dataset = validity
        else:
            return validity

        object_columns = dataset.select_dtypes(include=['object']).columns

        # Identify columns that might contain dates and convert them to datetime
        for column in object_columns:
            try:
                dataset[column] = pd.to_datetime(dataset[column])
            except (TypeError, ValueError):
                pass  # Ignore columns that cannot be converted t
            
        timestamp_columns = dataset.select_dtypes(include=['datetime64']).columns
        if timestamp_columns.any():
            primary_timestamp_column = timestamp_columns[0]
            dataset = dataset.set_index(primary_timestamp_column) 
            dataset.drop(columns=list(timestamp_columns[1:]), inplace=True)

        dataset.fillna(method="ffill", inplace=True)
        dataset = dataset.resample('M').sum()
        result = seasonal_decompose(dataset["Sales"], model='additive', period=5, extrapolate_trend='freq')
        print(result.trend.head())
        print(result.seasonal.head())
        print(result.resid.head())
        # Create a plot
        plt.figure(figsize=(10, 6))

        plt.plot(dataset.index, dataset["Sales"], label='Original')
        # Plot the trend component
        plt.plot(dataset.index, result.trend, label='Trend')

        # Plot the seasonal component
        plt.plot(dataset.index, result.seasonal, label='Seasonal')
        plt.title('Seasonal Component')
        plt.xlabel('Time')
        plt.ylabel('Seasonal')
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.legend()
        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert the BytesIO object to a base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Create a data URL from the base64 string
        image_data_url = 'data:image/png;base64,' + urllib.parse.quote(image_base64)
        return {"status": True, "seasonal_plot": image_data_url}


api.add_resource(FPsummary, "/fplot")
api.add_resource(SeasonalDistribution, "/seasonal-distribution")       

class TimeSeriesResource(Resource):
    def get(self):
        # Retrieve dataset from the dataset collection using user ID and dataset ID
        dataset_id = request.args['dataset_id']
        user_id=request.args['user_id']
        target=request.args['target']
        validity=check_validity(user_id,dataset_id)
        if isinstance(validity,pd.DataFrame):
            df=validity
        else:
            return validity

        # Load trained time series models from MongoDB
        model = TimeSeriesModel(data=df,target=target , session_id=123, db_name='model_db', collection_name='models')
        model.setup()
        model.compare_models(data_id=dataset_id)
        
        return {'message': 'Time series models trained and saved successfully'}, 200


class ClusteringResource(Resource):
    def get(self, user_id, dataset_id):
        # Retrieve dataset from the dataset collection using user ID and dataset ID
        dataset = dataset_collection.find_one({'user_id': user_id, 'dataset_id': dataset_id})
        
        if dataset is None:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Load trained clustering models from MongoDB
        model = ClusteringModel(data=dataset.data, session_id=123, db_name='model_db', collection_name='models')
        model.setup()
        model.compare_models()
        
        return jsonify({'message': 'Clustering models trained and saved successfully'}), 200


api.add_resource(TimeSeriesResource, '/timeseries')
api.add_resource(ClusteringResource, '/clustering')
if __name__ == '__main__':
    app.run(debug=True)
