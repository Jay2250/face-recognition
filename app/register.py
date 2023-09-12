from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
from app.utils import save_uploaded_image, save_face_features


register_face_router = APIRouter()


@register_face_router.post("/", response_model=dict)
async def register_face(file: UploadFile):
    try:
        # Saving the uploaded image
        file_path = save_uploaded_image(file)

        # Extracting facial features from the registered face
        registered_face = DeepFace.extract_faces(img_path=file_path)

        # Saving the facial features for recognition
        save_face_features(file.filename, registered_face)

        return JSONResponse(content={"message": "Face registered Successfully"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
