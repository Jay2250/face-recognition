from fastapi import FastAPI
from app.register import register_face_router
from app.recognize import recognize_face_router

app = FastAPI()

# Including routers from other modules
app.include_router(register_face_router, prefix="/register-face", tags=["Register"])
app.include_router(recognize_face_router, prefix="/recognize-face", tags=["Recognize"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    
