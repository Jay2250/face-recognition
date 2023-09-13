import json
import os
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from app.utils import save_uploaded_image
import face_recognition


recognize_face_router = APIRouter()

@recognize_face_router.post("/", response_model=dict)
async def recognize_face(file: UploadFile):
    try:
        # Saving the uploaded image temporarily
        file_path = save_uploaded_image(file)

        # Load the registered face image
        # registered_face_path = "registered_faces/woman-6109643_1280.jpg"
        registered_face_path = os.path.join("app", "registered_faces", f"{os.path.splitext(file.filename)[0]}.jpg")
        registered_face_image = face_recognition.load_image_file(registered_face_path)

        # Load the uploaded image
        uploaded_image = face_recognition.load_image_file(file_path)

        # Encoding faces in both images
        registered_face_encoding = face_recognition.face_encodings(registered_face_image)[0]
        uploaded_face_encodings = face_recognition.face_encodings(uploaded_image)

        # Check if any face is detected in the uploaded image
        if not uploaded_face_encodings:
            return JSONResponse(content={"recognized": False, "message": "No face detected in the image."})

        # Compare the detected face with the registered face
        results = face_recognition.compare_faces([registered_face_encoding], uploaded_face_encodings[0])

        # Check if the faces match
        recognized = results[0]

        # Convert recognized (boolean) to a string
        recognized_str = "Yes" if recognized else "No"

        return JSONResponse(content={"recognized": recognized_str})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
