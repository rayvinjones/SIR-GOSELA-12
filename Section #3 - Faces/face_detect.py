#pylint:disable=no-member

from pathlib import Path
import cv2 as cv

root = Path(__file__).resolve().parent.parent
image_dir = root / 'Resources' / 'Faces' / 'train' / 'Larianmar Pancho'
out_dir = root / 'Outputs' / 'Section3_LarianmarPancho'
out_dir.mkdir(parents=True, exist_ok=True)
xml_path = root / 'Section #3 - Faces' / 'haar_face.xml'

image_files = sorted(image_dir.glob('*.jpg'))
if not image_files:
    raise FileNotFoundError(f'No images found in: {image_dir}')

haar_cascade = cv.CascadeClassifier(str(xml_path))
if haar_cascade.empty():
    raise FileNotFoundError(f'Could not load Haar cascade: {xml_path}')

for image_path in image_files:
    img = cv.imread(str(image_path))
    if img is None:
        print(f'Skipping unreadable file: {image_path}')
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    print(f'{image_path.name}: {len(faces_rect)} faces found')

    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    output_path = out_dir / f'{image_path.stem}_detected.jpg'
    cv.imwrite(str(output_path), img)
    print(f'Saved: {output_path}')
