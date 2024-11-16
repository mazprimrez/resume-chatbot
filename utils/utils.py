import os
from datetime import datetime

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


# insert to database
class toDatabase:
    def __init__(self):
        self.client = None
        self.database_name = None
        self.collection_name = None
        self.uri = os.getenv("MONGODB_URI")

    def provision_pymongo(self):
        print('its provisioning!')
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        print('connected!')

    def store_to_database(self, query):
        if not self.client:
            self.provision_pymongo()

        db = self.client['personal_website']
        my_collections = db['feedback_message']

        current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        # Data yang ingin dimasukkan
        murid_1 = {'time':current_time,'message': query}

        my_collections.insert_one(murid_1)
    