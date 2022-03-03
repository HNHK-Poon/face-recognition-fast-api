from array import array
from time import time
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import face_recognition
import json
import numpy as np
import cv2
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from dbcontroller import Db_Controller

app = FastAPI()
db = Db_Controller("mongodb://localhost:27017/")

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Face Recognition API</title>
    </head>
</html>
"""

class registerData(BaseModel):
    name: str
    image: Optional[list] = []
    timestamp: int


class userResponse(BaseModel):
    name: str 
    embedding: str


@app.get("/")
async def home():
    return HTMLResponse(html)

@app.post("/register")
async def register(data: registerData, status_code=201):
    image = np.resize(data.image, (240,420,3))
    image = image.astype(np.uint8)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        face_location = np.asarray(face_locations[0])
        if len(face_locations) > 1:
            image_sizes = []
            for _face_location in face_locations:
                image_sizes.append(_face_location[2] - _face_location[0]) + (_face_location[1]-_face_location[3])
            face_location = face_locations[image_sizes.index(max(image_sizes))]
        else:
            face_location = np.asarray(face_locations[0])

        face_embeddings = face_recognition.face_encodings(image, face_locations, model="large")
        db.register_new_user(
            name=data.name,
            timestamp=data.timestamp,
            face_location=face_location,
            face_embedding=face_embeddings[0]
        )
        print(data.name, face_locations, face_embeddings, data.timestamp)

    return {"status": "success"}

@app.get("/getUser")
async def getUser(status_code=200, response_model=userResponse):
    cursor = users.find(
        {"name": "Alex"}, {"name":1, "embedding":1}
    )
    users_res = []
    for c in cursor:
        print(c, type(c))
        users_res.append({
            'name': c["name"],
            'embedding': c["embedding"]
        })
    return users_res


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("connected")
    while True:
        data = await websocket.receive_text()
        try:
            event = json.loads(data)
            event_type = event["eventType"]
            if event_type == "recognition":            
                image = np.resize(event["imageData"], (240,420,3))
                image = image.astype(np.uint8)
                face_locations = face_recognition.face_locations(image)
                face_embeddings = face_recognition.face_encodings(image, face_locations, model="large")
                matches = face_recognition.compare_faces(db.get_embeddings(), face_embeddings[0])
                if len(db.get_matches_indices(matches)) == 0 :
                    print("No matches")
                elif len(db.get_matches_indices(matches)) == 1:
                    print(matches, db.get_user_from_matches(matches))
                else:
                    face_distances = face_recognition.face_distance(db.get_embeddings(), face_embeddings[0])
                    best_match_index = np.argmin(face_distances)
                    print(best_match_index, face_distances)

                print(matches, db.get_user_from_matches(matches))
                reply = {
                    "eventType": "boundingBox",
                    "data": {
                        "faceLocations": face_locations[0],
                        "face_embeddings": ""
                    }
                }
                print(face_locations)
                await websocket.send_text(json.dumps(reply))
        except Exception as e:
            print(e)
