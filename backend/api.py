from flask import Flask, request, jsonify, url_for
from flask_restful import Resource, Api
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
import pickle
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
from pmdarima import auto_arima
from pmdarima.arima.utils import ndiffs
from scipy import stats
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns


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

# class Login(Resource):
#     def post(self):
#         user_data = request.get_json()
#         user_id = user_data.get('email')
#         password = user_data.get('password')

#         # Check if user_id and password are valid
#         # Save user data to MongoDB
#         if validate_credentials(email, password):

#             return {'message': 'Login successful'}, 200
#         else:
#             return {'message': 'Invalid credentials'}, 401
        






class Login(Resource):
    def post(self):
        user_data = request.get_json()
        email = user_data.get('email')
        password = user_data.get('password')

        # Check if user_id and password are valid
        # Save user data to MongoDB
        if validate_credentials(email, password):  
            return {'message': 'Login successful'}, 200
        else:
            return {'message': 'Invalid credentials'}, 401






def validate_credentials(email, password):
    # Retrieve the user data from the database based on the user_id
    user = user_data.find_one({'email': email})

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
        model = model_collection.find_one({'dataset_id': data_id,'model_type':"Timeseries"},{"model":1})
        if model:
            model = pickle.loads(model['model'])
        else:
            return {'error': 'Model not found'}, 404
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
            data = data.resample('M').sum()
            data.index = pd.to_datetime(data.index)

        data.fillna(method="ffill", inplace=True)
        data.index = data.index.to_period("M")

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

        actual_data = {str(index): value for index, value in zip(data.index[-n:], data[target][-n:])}

        forecast_data = pd.DataFrame(index=range(len(data), len(data) + n))
        # Use ARIMA for forecasting
        y = data[target].tail(n)
        model=model['model']
        y_pred = model.predict(n_periods=n,X=y)
        print(y_pred.describe())
        summary_stats = {"mean": y_pred.mean(), "std": y_pred.std()}
        print(summary_stats)

        # Prepare forecast data for the frontend
        forecast_data = {str(index): value for index, value in zip(range(len(data), len(data) + n), y_pred)}
        print(y.index,y_pred.index)
        # Create a plot
        y.index = y.index.to_timestamp()
        y_pred.index = y_pred.index.to_timestamp()
        plt.figure(figsize=(10, 6))
        plt.plot(y.index, y.values, label="Actual", color="blue")
        # Plot forecast data
        plt.plot(y_pred.index, y_pred.values, label="Forecast", color="orange")
        plt.title('Forecast')
        plt.xlabel('Time')
        plt.ylabel(target)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%M'))
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

        return {"status": True, "summary_stats": summary_stats, "actual_data": actual_data,
                "forecast_data": forecast_data, "forecast_plot": image_data_url}

api.add_resource(FPsummary, "/fplot")

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

api.add_resource(SeasonalDistribution, "/seasonal-distribution")   


def sales_forecasting(data, target_column):
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
        data.index = pd.to_datetime(data.index)

    data.fillna(method="ffill", inplace=True)
    data.index = data.index.to_period("D")

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
    print(data.head())
    # Train auto ARIMA model
    try:
        model = auto_arima(data[target_column], suppress_warnings=True)
        model.fit(data[target_column])
    except:
        return {"status": "Error", "error": "Error in training auto ARIMA model"}

    return {"status": "success", "model": model}

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

        model =sales_forecasting(data=df,target_column=target)
        print(model)
        if model["status"]=="success":
            model=pickle.dumps(model)
            
            model_record = model_collection.find_one({'model_type': "Timeseries", 'dataset_id': dataset_id})
            if model_record is not None:
                print("here")
                model_collection.update_one(
                    {'model_type': "Timeseries", 'dataset_id': dataset_id},
                    {'$set': { 'model': model}}
                )
            else:
                model_collection.insert_one({'model_type': "Timeseries", 'model': model, 'dataset_id': dataset_id, 'target': target})

            return {'message': 'Time series models trained and saved successfully'}, 200

        else:
            return model,400


class ClusteringResource(Resource):
    def get(self):
        data=request.get_json()
        data_id = data['data_id']
        n=int(data['n_periods'])
        user_id=data['user_id']
        target=data['target']

        validity=check_validity(user_id,data_id)
        if isinstance(validity,pd.DataFrame):
            data=validity
        else:
            return validity

        
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)

            # Use the elbow method to find the optimal number of clusters
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                kmeans.fit(scaled_data)
                wcss.append(kmeans.inertia_)

            # Plot the elbow method
            plt.plot(range(1, 11), wcss, marker='o')
            plt.title('Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
            plt.show()

            # Based on the elbow method, choose the optimal number of clusters
            # Replace 'optimal_clusters' with the number you choose
            optimal_clusters = 3

            # Perform K-means clustering with the optimal number of clusters
            kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
            data['Cluster'] = kmeans.fit_predict(scaled_data)

            # Visualize the clusters using pair plot
            sns.pairplot(data, hue='Cluster')
            plt.suptitle('Pair Plot with Clusters', y=1.02)
            plt.show()

            # Analyze the characteristics of each cluster
            cluster_stats = data.groupby('Cluster').mean()
            print("Cluster Statistics:")
            print(cluster_stats)
            
api.add_resource(TimeSeriesResource, '/timeseries')
api.add_resource(ClusteringResource, '/clustering')
if __name__ == '__main__':
    app.run(debug=True)
