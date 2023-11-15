import unittest
import pandas as pd
from models import BaseModel
from pymongo import MongoClient
from bson.binary import Binary
import pickle
import os

class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.base_model = BaseModel(123, 'test_db', 'test_collection')
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['test_db']
        self.collection = self.db['test_collection']

    def test_load_data_csv(self):
        # Create a test CSV file
        df = pd.DataFrame({'target': [1, 2, 3]})
        df.to_csv('test.csv', index=False)

        # Load data
        self.base_model.load_data('test.csv', 'target', 'test_data')

        # Check if data is loaded correctly
        pd.testing.assert_frame_equal(self.base_model.data, df)

        # Check if data is saved to MongoDB
        record = self.collection.find_one({'data_id': 'test_data'})
        self.assertIsNotNone(record)
        self.assertEqual(pickle.loads(record['data']), df)

        # Clean up
        os.remove('test.csv')

    def test_load_data_excel(self):
        # Create a test Excel file
        df = pd.DataFrame({'target': [1, 2, 3]})
        df.to_excel('test.xlsx', index=False)

        # Load data
        self.base_model.load_data('test.xlsx', 'target', 'test_data')

        # Check if data is loaded correctly
        pd.testing.assert_frame_equal(self.base_model.data, df)

        # Check if data is saved to MongoDB
        record = self.collection.find_one({'data_id': 'test_data'})
        self.assertIsNotNone(record)
        self.assertEqual(pickle.loads(record['data']), df)

        # Clean up
        os.remove('test.xlsx')

    def test_load_data_invalid_file_type(self):
        # Load data
        self.base_model.load_data('test.txt', 'target', 'test_data')

        # Check if data is not loaded
        self.assertIsNone(self.base_model.data)

    def test_load_data_invalid_target(self):
        # Create a test CSV file
        df = pd.DataFrame({'target': [1, 2, 3]})
        df.to_csv('test.csv', index=False)

        # Load data
        self.base_model.load_data('test.csv', 'invalid_target', 'test_data')

        # Check if data is not loaded
        self.assertIsNone(self.base_model.data)

        # Clean up
        os.remove('test.csv')

if __name__ == '__main__':
    unittest.main()