from array import array
from time import time
from dbcontroller import DATA_CREATED_AT, DATA_EMPLOYEE_ID, DATA_NAME, DATA_POSITION, DATA_TIMESTAMP
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import face_recognition
import json
import numpy as np
import cv2
from pydantic import BaseModel
from typing import List, Optional
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
    <body>
        <h1> Welcome </h1>
    </body>
</html>
"""

class registerData(BaseModel):
    name: str
    position: str
    employeeId: str
    createAt: str
    imageArray: List[int]
    timestamp: int


class userResponse(BaseModel):
    name: str 
    embedding: str


@app.get("/")
async def home():
    return HTMLResponse(html)

@app.post("/register")
def register(data:registerData, status_code=201):
    image = np.resize(data.imageArray, (240,420,3))
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
        # db.register_new_user(
        #     name=data.name,
        #     timestamp=data.timestamp,
        #     face_location=face_location,
        #     face_embedding=face_embeddings[0]
        # )
        print(data.name, data.id, data.position, data.createAt, face_location, face_embeddings, data.timestamp)

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
                if len(face_locations) > 0:
                    face_location = np.asarray(face_locations[0])
                    if len(face_locations) > 1:
                        image_sizes = []
                        for _face_location in face_locations:
                            image_sizes.append(_face_location[2] - _face_location[0]) + (_face_location[1]-_face_location[3])
                        face_location = face_locations[image_sizes.index(max(image_sizes))]
                    else:
                        face_location = face_locations[0]
                    
                    face_embeddings = face_recognition.face_encodings(image, [face_location], model="large")
                    matches = face_recognition.compare_faces(db.get_embeddings(), face_embeddings[0])
                    if len(db.get_matches_indices(matches)) == 0 :
                        print("No matches")
                    elif len(db.get_matches_indices(matches)) == 1:
                        reply = {
                        "eventType": "boundingBox",
                        "data": {
                            "faceLocations": face_locations[0],
                            "face_embeddings": "",
                            "user": db.get_user_from_matches(matches)
                        }
                    }
                    else:
                        face_distances = face_recognition.face_distance(db.get_embeddings(), face_embeddings[0])
                        best_match_index = np.argmin(face_distances)
                        reply = {
                            "eventType": "boundingBox",
                            "data": {
                                "faceLocations": face_locations[0],
                                "face_embeddings": "",
                                "user": db.get_user_from_index(best_match_index)
                            }
                        }
                        
                    print(reply)
                    await websocket.send_text(json.dumps(reply))
                else: 
                    print("No user found")

            elif event_type == "register":
                data = event["data"]
                image = np.resize(data["imageArray"], (240,420,3))
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
                        name=data[DATA_NAME],
                        position=data[DATA_POSITION],
                        employeeId=data[DATA_EMPLOYEE_ID],
                        createdAt=data[DATA_CREATED_AT],
                        timestamp=data[DATA_TIMESTAMP],
                        face_embedding=face_embeddings[0]
                    )

                print(face_embeddings, data[DATA_NAME], data[DATA_POSITION], data[DATA_EMPLOYEE_ID], data[DATA_CREATED_AT], data[DATA_TIMESTAMP])
                

        except Exception as e:
            print(e)
