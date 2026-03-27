import os
import cv2
import face_recognition
import numpy as np

KNOWN_DIR = os.path.join(os.path.dirname(__file__), "..", "known_people")
TOLERANCE = 0.45
MODEL = "hog"  # "hog" (CPU) ya "cnn" (GPU) 


def load_known_faces():
    known_faces = []
    known_names = []

    if not os.path.exists(KNOWN_DIR):
        raise FileNotFoundError(f"Known people folder nahi mila: {KNOWN_DIR}")

    for filename in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, filename)
        if not os.path.isfile(path):
            continue
        name, ext = os.path.splitext(filename)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"Warning: koi face encoding nahi mila: {filename}")
            continue

        known_faces.append(encodings[0])
        known_names.append(name)

    return known_faces, known_names


def match_face(query_image_path):
    image = cv2.imread(query_image_path)
    if image is None:
        raise FileNotFoundError(f"Query image nahi mili: {query_image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb, model=MODEL)
    query_encodings = face_recognition.face_encodings(rgb, face_locations)

    known_faces, known_names = load_known_faces()

    if not known_faces:
        raise ValueError("Known faces load nahi hue. known_people folder check karo.")

    matches = []
    for (top, right, bottom, left), query_encoding in zip(face_locations, query_encodings):
        distances = face_recognition.face_distance(known_faces, query_encoding)
        best_idx = np.argmin(distances)
        name = "Unknown" if distances[best_idx] > TOLERANCE else known_names[best_idx]

        matches.append({
            "name": name,
            "distance": float(distances[best_idx]),
            "location": (top, right, bottom, left)
        })

    return image, matches


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python face_match.py <query_image>")
        sys.exit(1)

    query_image = sys.argv[1]
    image, matches = match_face(query_image)

    print(f"Detected {len(matches)} face(s) in query image")
    for m in matches:
        print(m)

    for m in matches:
        top, right, bottom, left = m["location"]
        color = (0, 255, 0) if m["name"] != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, f"{m['name']} {m['distance']:.2f}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite("face_match_result.jpg", image)
    print("Result saved: face_match_result.jpg")
