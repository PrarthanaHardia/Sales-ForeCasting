from flask import Flask, request, jsonify, url_for
from flask_restful import Resource, Api
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
from pycaret.clustering import ClusteringExperiment
# from pycaret.internal.pycaret_experiment import TimeSeriesExperiment, ClusteringExperiment
import pickle
from bson.binary import Binary
from werkzeug.utils import secure_filename
from datetime import datetime
import matplotlib.pyplot as plt
import uuid

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
                file_url = url_for("get_dataset", file_id=file_id, _external=True)

                data = {
                    "time_stamp":time_stamp,
                    "user_id": user_id,
                    "file_id": file_id,
                    "content_type": content_type,
                    "dataset_url": file_url,
                    "model_url":None,
                    "model_id":None
                }

                user_data.delete_one({"user_id": user_id})
                user_data.insert_one(data)

                return jsonify({
                    "status": True,
                    "message": "File uploaded",
                    "file_id": file_id,
                    "file_url": file_url  
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

    def save_model(self, model_type, best_model, dataset_id):
        pickled_model = Binary(pickle.dumps(best_model))
        self.collection.insert_one({'model_type': model_type, 'model': pickled_model, 'dataset_id': dataset_id})

    def load_model(self, model_type, dataset_id):
        record = self.collection.find_one({'model_type': model_type, 'dataset_id': dataset_id})
        if record is not None:
            return pickle.loads(record['model'])
        else:
            return None

class TimeSeriesModel(BaseModel):
    def setup(self):
        self.ts_exp = TSForecastingExperiment()
        self.ts_exp.setup(data=self.data, target=self.target, session_id=self.session_id)

    def compare_models(self):
        self.model = self.ts_exp.compare_models()
        self.save_model('Time_Series', self.model, self.data_id)


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


class ClusteringModel(BaseModel):
    def setup(self):
        self.clustering_exp = ClusteringExperiment()
        self.clustering_exp.setup(data=self.data, session_id=self.session_id)

    def compare_models(self):
        self.best_model = self.clustering_exp.create_model("kmeans")
        self.save_model('KMeans_Clustering', self.best_model)

class TimeSeriesResource(Resource):
    def get(self, user_id, dataset_id):
        # Retrieve dataset from the dataset collection using user ID and dataset ID
        dataset = dataset_collection.find_one({'user_id': user_id, 'dataset_id': dataset_id})
        
        if dataset is None:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Load trained time series models from MongoDB
        model = TimeSeriesModel(data=dataset.data, target=dataset.target, session_id=123, db_name='model_db', collection_name='models')
        model.setup()
        model.compare_models()
        
        return jsonify({'message': 'Time series models trained and saved successfully'}), 200


class ClusteringResource(Resource):
    def get(self, user_id, dataset_id):
        # Retrieve dataset from the dataset collection using user ID and dataset ID
        dataset = dataset_collection.find_one({'user_id': user_id, 'dataset_id': dataset_id})
        
        if dataset is None:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Load trained clustering models from MongoDB
        model = ClusteringModel(data=dataset.data, target=dataset.target, session_id=123, db_name='model_db', collection_name='models')
        model.setup()
        model.compare_models()
        
        return jsonify({'message': 'Clustering models trained and saved successfully'}), 200


api.add_resource(TimeSeriesResource, '/timeseries')
api.add_resource(ClusteringResource, '/clustering')
if __name__ == '__main__':
    app.run(debug=True)
