# pylint:disable=no-member

from pathlib import Path
import cv2 as cv
import numpy as np

# Project root
root = Path(__file__).resolve().parent.parent

# Training images folder
DIR = root / 'Resources' / 'Faces' / 'train'

# Haar Cascade
xml_path = root / 'Section #3 - Faces' / 'haar_face.xml'
haar_cascade = cv.CascadeClassifier(str(xml_path))

# People (must match the folder names exactly)
people = [
    'Ben Afflek',
    'Elton John',
    'Jerry Seinfield',
    'Diether Flores',
    'Madonna',
    'Mindy Kaling'
]

features = []
labels = []


def create_train():
    for person in people:

        path = DIR / person
        label = people.index(person)

        if not path.exists():
            print(f"Folder not found: {path}")
            continue

        for img_path in path.iterdir():

            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            img_array = cv.imread(str(img_path))

            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4
            )

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]

                features.append(faces_roi)
                labels.append(label)


create_train()

print("Training completed!")

features = np.array(features, dtype="object")
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

model_path = root / 'Section #3 - Faces' / 'face_trained.yml'

face_recognizer.save(str(model_path))

np.save(root / 'Section #3 - Faces' / 'features.npy', features)
np.save(root / 'Section #3 - Faces' / 'labels.npy', labels)

print("Model saved successfully!")
print(f"Total training images: {len(features)}")