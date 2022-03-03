import pymongo
import numpy as np
import json


DATA_NAME = "name"
DATA_FACE_LOCATION = "location"
DATA_FACE_EMBEDDING = "embedding"
DATA_TIMESTAMP = "timestamp"

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

class UserCache:
    DATA_NAME = "name"
    DATA_FACE_LOCATION = "face_location"
    DATA_FACE_EMBEDDING = "face_embedding"
    DATA_TIMESTAMP = "timestamp"

    def __init__(self) -> None:
        self.user_data = []
        self.embeddings = []
        self.users = []

    def update(self, user_data):
        self.user_data = user_data
        for user in self.user_data:
            self.embeddings.append(json.loads(user[DATA_FACE_EMBEDDING]))
            self.users.append(user[DATA_NAME])

    def get(self):
        return self.user_embeddings


class Db_Controller:
    DB_NAME = "face-recognition-db"
    USER_COLLECTION = "user"
    RECORD_COLLECTION = "record"

    def __init__(
        self, 
        path="mongodb://localhost:27017/"
     ) -> None:
        self.mongo_client = pymongo.MongoClient(path)
        self.db = None
        self.user_collection = None
        self.record_collection = None
        self.user_cache = UserCache()
        self.init_db_collections()
        self.update_user_cache()

    def init_db_collections(self):
        self.db = self.mongo_client[self.DB_NAME]
        self.user_collection = self.db[self.USER_COLLECTION]
        self.record_collection = self.db[self.RECORD_COLLECTION]
        print("Successfully init db, list of collections:", self.mongo_client.list_database_names())

    def register_new_user(
        self, 
        name, 
        timestamp, 
        face_location, 
        face_embedding
    ):
        self.user_collection.insert_one({
            DATA_NAME: name,
            DATA_FACE_EMBEDDING: json.dumps(face_embedding, cls=NumpyArrayEncoder),
            DATA_FACE_LOCATION: json.dumps(face_location, cls=NumpyArrayEncoder),
            DATA_TIMESTAMP: timestamp
        })
        self.update_user_cache()

    def get_all_users(self):
        return self.user_cache.get()

    def update_user_cache(self):
        self.user_cache.update(
            self.user_collection.find(
                {}
            )
        )

    def get_embeddings(self):
        return self.user_cache.embeddings

    def get_matches_indices(self, matches):
        matches_indices = [i for i, val in enumerate(matches) if val]
        return matches_indices

    def get_user_from_matches(self, matches):
        matches_users = [i for (i, v) in zip(self.user_cache.users, matches) if v]
        return matches_users

