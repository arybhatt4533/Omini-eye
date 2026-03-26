import os
from face_detection.face_detect import detect_faces
from face_matching.face_match import match_face


def ensure_known_folder():
    if not os.path.exists("known_people"):
        os.makedirs("known_people")
        print("created known_people folder, ab waha images daalo (name.jpg)")


def main():
    ensure_known_folder()
    choice = input("1) Face detect 2) Face match 3) Exit\nChoose: ").strip()

    if choice == "1":
        image_path = input("Image path for detection (e.g., query.jpg): ").strip()
        image, faces = detect_faces(image_path)
        print(f"{len(faces)} face(s) detected")

    elif choice == "2":
        image_path = input("Image path for matching (e.g., query.jpg): ").strip()
        image, matches = match_face(image_path)
        print(f"{len(matches)} face(s) matched")

    else:
        print("Exit")


if __name__ == "__main__":
    main()
