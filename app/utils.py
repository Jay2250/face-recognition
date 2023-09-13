import json
import os
from deepface import DeepFace
import numpy as np


def save_uploaded_image(file):
    file_path = os.path.join("app", "registered_faces", file.filename)
    with open(file_path, "wb") as image_file:
        image_file.write(file.file.read())
    return file_path


def save_face_features(filename, face_features_list):
    try:
        if not isinstance(face_features_list, list):
            raise ValueError("Face features should be a list of dictionaries.")

        # Convert NumPy arrays to lists in each dictionary
        for face_features in face_features_list:
            for key, value in face_features.items():
                if isinstance(value, np.ndarray):
                    face_features[key] = value.tolist()

        features_path = os.path.join("app", "registered_faces", f"{os.path.splitext(filename)[0]}.json")
        with open(features_path, "w") as features_file:
            json.dump(face_features_list, features_file)
    except Exception as e:
        raise ValueError(f"Error saving face features: {str(e)}")




def load_face_features(features_path):
    try:
        with open(features_path, "r") as features_file:
            face_features_list = json.load(features_file)

        # Ensure face_features_list is a list of dictionaries
        if not isinstance(face_features_list, list) or not all(isinstance(item, dict) for item in face_features_list):
            raise ValueError("Invalid format for face features.")

        return face_features_list
    except Exception as e:
        raise ValueError(f"Error loading face features: {str(e)}")
