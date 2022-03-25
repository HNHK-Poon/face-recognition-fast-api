import pymongo
import numpy as np
import json
import time


DATA_NAME = "name"
DATA_FACE_LOCATION = "location"
DATA_POSITION = "position"
DATA_EMPLOYEE_ID = "employeeId"
DATA_CREATED_AT = "createdAt"
DATA_FACE_EMBEDDING = "embedding"
DATA_TIMESTAMP = "timestamp"
DATA_IMAGE_64 = "image64"

# Encode numpy array in general type
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

# Cache as a class to holds all the user information, e.g. user data, user embedding
class UserCache:
    DATA_NAME = "name"
    DATA_FACE_LOCATION = "face_location"
    DATA_FACE_EMBEDDING = "face_embedding"
    DATA_TIMESTAMP = "timestamp"

    # init cache with empty array
    def __init__(self) -> None:
        self.user_data = []
        self.embeddings = []
        self.users = []

    # update cache from latest retrieved data
    def update(self, user_data):
        self.user_data = user_data
        self.embeddings = []
        self.users = []
        for user in self.user_data:
            self.embeddings.append(json.loads(user[DATA_FACE_EMBEDDING]))
            user.pop(DATA_FACE_EMBEDDING)
            user.pop("_id")
            self.users.append(user)

    # return user embeddings
    def get(self):
        return self.user_embeddings


# Controller class that handles all the db functions
class Db_Controller:
    DB_NAME = "face-recognition-db"
    USER_COLLECTION = "user"
    RECORD_COLLECTION = "record"

    # mongo db initialization
    def __init__(
        self,
        path="mongodb://mongodb2"
    ) -> None:
        self.mongo_client = pymongo.MongoClient(path)
        self.db = None
        self.user_collection = None
        self.record_collection = None
        self.user_cache = UserCache()
        time.sleep(2)
        self.init_db_collections()
        self.update_user_cache()

    # create db, collection if not exist
    def init_db_collections(self):
        self.db = self.mongo_client[self.DB_NAME]
        self.user_collection = self.db[self.USER_COLLECTION]
        self.record_collection = self.db[self.RECORD_COLLECTION]

    # Insert new user
    def register_new_user(
        self,
        name,
        position,
        employeeId,
        createdAt,
        timestamp,
        image64,
        face_embedding
    ):
        self.user_collection.insert_one({
            DATA_NAME: name,
            DATA_POSITION: position,
            DATA_EMPLOYEE_ID: employeeId,
            DATA_CREATED_AT: createdAt,
            DATA_TIMESTAMP: timestamp,
            DATA_IMAGE_64: image64,
            DATA_FACE_EMBEDDING: json.dumps(face_embedding, cls=NumpyArrayEncoder),
        })
        self.update_user_cache()

    # Remove user
    def remove_user(
        self,
        employeeId
    ):
        self.user_collection.delete_one({DATA_EMPLOYEE_ID: employeeId})

    # Get all user's data from cache
    def get_all_users(self):
        return self.user_cache.users

    # Update cache
    def update_user_cache(self):
        self.user_cache.update(
            self.user_collection.find(
                {}
            )
        )

    # Get all user embeddings
    def get_embeddings(self):
        return self.user_cache.embeddings

    # Return indices for matched users
    def get_matches_indices(self, matches):
        matches_indices = [i for i, val in enumerate(matches) if val]
        return matches_indices

    # Return user's data for matched indices
    def get_user_from_matches(self, matches):
        matches_user = [i for (i, v) in zip(
            self.user_cache.users, matches) if v]
        return matches_user[0]

    # Get user's from best matched index
    def get_user_from_index(self, index):
        matches_user = self.user_cache.users[index]
        return matches_user
