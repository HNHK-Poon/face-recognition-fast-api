from array import array
from time import time
from turtle import pos
from recognition import Vision
from dbcontroller import DATA_CREATED_AT, DATA_EMPLOYEE_ID, DATA_NAME, DATA_POSITION, DATA_TIMESTAMP, DATA_IMAGE_64
from fastapi import FastAPI, WebSocket, Request, status
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import json
import numpy as np
import cv2
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from dbcontroller import Db_Controller

# Init mongodb contoller and recognition algorithms
app = FastAPI()
db = Db_Controller("mongodb://face-recognition-mongodb")
# db = Db_Controller("mongodb://localhost:27017/")
vision = Vision(tolerance=0.4)

# Enable CORS from recognised origin
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
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


class userResponse(BaseModel):
    name: str
    embedding: str


@app.get("/")
async def home():
    return HTMLResponse(html)


class registerData(BaseModel):
    name: str
    position: str
    employeeId: str
    createdAt: str
    image64: str
    imageArray: List[int]
    timestamp: int


@app.post("/register")
def register(data: registerData, status_code=201):

    # Resize and reformat image from (302400, 1) to (240, 420, 3)
    image = np.resize(data.imageArray, (240, 420, 3))
    image = image.astype(np.uint8)

    # Get all face's locations
    face_locations = vision.face_locations(image)
    if len(face_locations) > 0:
        face_location = np.asarray(face_locations[0])

        # Choose the largest face if there is more than 1 face
        if len(face_locations) > 1:
            image_sizes = []
            for _face_location in face_locations:
                image_sizes.append(
                    _face_location[2] - _face_location[0]) + (_face_location[1]-_face_location[3])
            face_location = face_locations[image_sizes.index(max(image_sizes))]
        else:
            face_location = np.asarray(face_locations[0])

        # Compute face embedding from face image
        face_embeddings = vision.face_encodings(
            image, [face_location])

        # Save new user into database
        db.register_new_user(
            name=data.name,
            position=data.position,
            employeeId=data.employeeId,
            createdAt=data.createdAt,
            timestamp=data.timestamp,
            image64=data.image64,
            face_embedding=face_embeddings[0]
        )

    return {"status": "success"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get("/getUsers")
async def getUser(status_code=200):
    users = db.get_all_users()
    return users


class deleteData(BaseModel):
    employeeId: str


@app.post("/deleteUser")
def register(data: deleteData, status_code=201):

    # Remove user from database
    db.remove_user(employeeId=data.employeeId)
    db.update_user_cache()
    return {"status": "success"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    # Listening to incoming websocket events
    await websocket.accept()
    while True:

        # Receive incoming websocket events
        data = await websocket.receive_text()
        try:
            event = json.loads(data)
            event_type = event["eventType"]

            # Check if incoming event is for recognition
            if event_type == "recognition":

                # Resize and reformat image from (302400, 1) to (240, 420, 3)
                image = np.resize(event["imageData"], (240, 420, 3))
                image = image.astype(np.uint8)

                # Get all face's locations
                face_locations = vision.face_locations(image)
                if len(face_locations) > 0:
                    face_location = np.asarray(face_locations[0])

                    # Choose the largest face if there is more than 1 face
                    if len(face_locations) > 1:
                        image_sizes = []
                        for _face_location in face_locations:
                            image_sizes.append(
                                _face_location[2] - _face_location[0]) + (_face_location[1]-_face_location[3])
                        face_location = face_locations[image_sizes.index(
                            max(image_sizes))]
                    else:
                        face_location = face_locations[0]

                    # Compute face embedding from face image
                    face_embeddings = vision.face_encodings(
                        image, [face_location])

                    # Match face embedding with embeddings from database
                    matches = vision.compare_faces(
                        db.get_embeddings(), face_embeddings[0])

                    # If there is no face matched, return userNotDetected
                    if len(db.get_matches_indices(matches)) == 0:
                        reply = {
                            "eventType": "userNotDetected",
                            "data": {
                                "faceLocations": face_location,
                            }
                        }
                    # If there is only one matched face, return user detected
                    elif len(db.get_matches_indices(matches)) == 1:
                        reply = {
                            "eventType": "userDetected",
                            "data": {
                                "faceLocations": face_location,
                                "face_embeddings": "",
                                "user": db.get_user_from_matches(matches)
                            }
                        }
                    # If there are more than one matched face, return the highest confidence result
                    else:
                        face_distances = vision.face_distance(
                            db.get_embeddings(), face_embeddings[0])
                        best_match_index = np.argmin(face_distances)
                        reply = {
                            "eventType": "multiUserDetected",
                            "data": {
                                "faceLocations": face_location,
                                "face_embeddings": "",
                                "user": db.get_user_from_index(best_match_index)
                            }
                        }

                    # Send event to client through socket
                    await websocket.send_text(json.dumps(reply))

                # If no face detected, Send faceNotDetected event to client through socket
                else:
                    reply = {
                        "eventType": "faceNotDetected",
                    }
                    await websocket.send_text(json.dumps(reply))

        except Exception as e:
            print(e)
