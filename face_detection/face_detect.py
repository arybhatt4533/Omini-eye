import cv2
import face_recognition

MODEL = "hog"  # "hog" (CPU) ya "cnn" (GPU accelerator)


def detect_faces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image nahi mili: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb, model=MODEL)
    results = []
    for (top, right, bottom, left) in face_locations:
        results.append({
            "top": top,
            "right": right,
            "bottom": bottom,
            "left": left
        })

    return image, results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python face_detect.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    img, faces = detect_faces(img_path)
    print(f"{len(faces)} face(s) detected")
    for i, face in enumerate(faces, 1):
        print(f"#{i}: {face}")

    for face in faces:
        cv2.rectangle(img,
                      (face["left"], face["top"]),
                      (face["right"], face["bottom"]),
                      (0, 255, 0), 2)

    out_path = "face_detected.jpg"
    cv2.imwrite(out_path, img)
    print(f"Output saved: {out_path}")
