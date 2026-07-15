# pylint:disable=no-member

from pathlib import Path
import random
import cv2 as cv

root = Path(__file__).resolve().parent.parent
base_dir = root / 'Section #3 - Faces'

# Image to recognize
image_path = root / 'Resources' / 'Faces' / 'val' / 'Diether Flores' / 'diether2.jpg'

xml_path = base_dir / 'haar_face.xml'
model_path = base_dir / 'face_trained.yml'

people = [
    'Ben Afflek',
    'Elton John',
    'Jerry Seinfield',
    'Diether Flores',
    'Madonna',
    'Mindy Kaling'
]

haar_cascade = cv.CascadeClassifier(str(xml_path))

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(str(model_path))

img = cv.imread(str(image_path))

if img is None:
    raise FileNotFoundError(f'Could not read image: {image_path}')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:

    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)

    person = people[label]

    print(f'Label = {person}')
    print(f'Confidence = {confidence}')

    cv.putText(img, person, (20, 35),
               cv.FONT_HERSHEY_COMPLEX,
               1,
               (0, 255, 0),
               2)

    cv.rectangle(img,
                 (x, y),
                 (x+w, y+h),
                 (0, 255, 0),
                 2)

    # If it is Larianmar Pancho, display a random image
    if person == "Larianmar Pancho":

        image_folder = root / 'Resources' / 'Faces' / 'train' / 'Larianmar Pancho'

        image_files = []

        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(image_folder.glob(ext))

        if image_files:

            random_image = random.choice(image_files)

            print("Showing:", random_image.name)

            random_img = cv.imread(str(random_image))

            if random_img is not None:

                # Resize while keeping aspect ratio
                max_width = 500
                max_height = 500

                h, w = random_img.shape[:2]

                scale = min(max_width / w, max_height / h)

                new_width = int(w * scale)
                new_height = int(h * scale)

                random_img = cv.resize(random_img, (new_width, new_height))

                cv.imshow("Random Image", random_img)

# Save recognition result
output_path = root / 'Outputs' / 'Section3_recognition_result.jpg'
output_path.parent.mkdir(parents=True, exist_ok=True)

cv.imwrite(str(output_path), img)

print(f'Saved: {output_path}')

# Resize recognition image before displaying
max_width = 500
max_height = 500

h, w = img.shape[:2]
scale = min(max_width / w, max_height / h)

new_width = int(w * scale)
new_height = int(h * scale)

img = cv.resize(img, (new_width, new_height))

# Display recognition result
cv.imshow("Recognition Result", img)

cv.waitKey(0)
cv.destroyAllWindows()