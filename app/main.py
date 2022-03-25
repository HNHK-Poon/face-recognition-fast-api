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

app = FastAPI()
db = Db_Controller("mongodb://localhost:27017/")
vision = Vision(tolerance=0.4)

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
    createdAt: str
    image64: str
    imageArray: List[int]
    timestamp: int


class userResponse(BaseModel):
    name: str
    embedding: str


class deleteData(BaseModel):
    employeeId: str


@app.get("/")
async def home():
    return HTMLResponse(html)


@app.post("/register")
def register(data: registerData, status_code=201):
    print(data)
    image = np.resize(data.imageArray, (240, 420, 3))
    image = image.astype(np.uint8)
    face_locations = vision.face_locations(image)
    if len(face_locations) > 0:
        face_location = np.asarray(face_locations[0])
        if len(face_locations) > 1:
            image_sizes = []
            for _face_location in face_locations:
                image_sizes.append(
                    _face_location[2] - _face_location[0]) + (_face_location[1]-_face_location[3])
            face_location = face_locations[image_sizes.index(max(image_sizes))]
        else:
            face_location = np.asarray(face_locations[0])

        face_embeddings = vision.face_encodings(
            image, face_locations)
        db.register_new_user(
            name=data.name,
            position=data.position,
            employeeId=data.employeeId,
            createdAt=data.createdAt,
            timestamp=data.timestamp,
            image64=data.image64,
            face_embedding=face_embeddings[0]
        )
        print(data.name, data.employeeId, data.position, data.createdAt,
              face_location, face_embeddings, data.timestamp)

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


@app.post("/deleteUser")
def register(data: deleteData, status_code=201):
    print(data, data.employeeId)
    db.remove_user(employeeId=data.employeeId)
    db.update_user_cache()
    return {"status": "success"}


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
                image = np.resize(event["imageData"], (240, 420, 3))
                image = image.astype(np.uint8)
                face_locations = vision.face_locations(image)
                if len(face_locations) > 0:
                    face_location = np.asarray(face_locations[0])
                    if len(face_locations) > 1:
                        image_sizes = []
                        for _face_location in face_locations:
                            image_sizes.append(
                                _face_location[2] - _face_location[0]) + (_face_location[1]-_face_location[3])
                        face_location = face_locations[image_sizes.index(
                            max(image_sizes))]
                    else:
                        face_location = face_locations[0]

                    face_embeddings = vision.face_encodings(
                        image, [face_location])
                    matches = vision.compare_faces(
                        db.get_embeddings(), face_embeddings[0])
                    if len(db.get_matches_indices(matches)) == 0:
                        reply = {
                            "eventType": "userNotDetected",
                            "data": {
                                "faceLocations": face_location,
                            }
                        }
                    elif len(db.get_matches_indices(matches)) == 1:
                        reply = {
                            "eventType": "userDetected",
                            "data": {
                                "faceLocations": face_location,
                                "face_embeddings": "",
                                "user": db.get_user_from_matches(matches)
                            }
                        }
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
                    await websocket.send_text(json.dumps(reply))
                else:
                    reply = {
                        "eventType": "faceNotDetected",
                    }
                    await websocket.send_text(json.dumps(reply))

            elif event_type == "register":
                data = event["data"]
                image = np.resize(data["imageArray"], (240, 420, 3))
                image = image.astype(np.uint8)
                face_locations = vision.face_locations(image)
                if len(face_locations) > 0:
                    face_location = np.asarray(face_locations[0])
                    if len(face_locations) > 1:
                        image_sizes = []
                        for _face_location in face_locations:
                            image_sizes.append(
                                _face_location[2] - _face_location[0]) + (_face_location[1]-_face_location[3])
                        face_location = face_locations[image_sizes.index(
                            max(image_sizes))]
                    else:
                        face_location = np.asarray(face_locations[0])

                    face_embeddings = vision.face_encodings(
                        image, face_locations)

                    db.register_new_user(
                        name=data[DATA_NAME],
                        position=data[DATA_POSITION],
                        employeeId=data[DATA_EMPLOYEE_ID],
                        createdAt=data[DATA_CREATED_AT],
                        timestamp=data[DATA_TIMESTAMP],
                        image64=data[DATA_IMAGE_64],
                        face_embedding=face_embeddings[0]
                    )

                print(face_embeddings, data[DATA_NAME], data[DATA_POSITION],
                      data[DATA_EMPLOYEE_ID], data[DATA_CREATED_AT], data[DATA_TIMESTAMP])

        except Exception as e:
            print(e)
