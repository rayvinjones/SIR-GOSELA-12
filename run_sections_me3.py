import os
from pathlib import Path
import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / 'Resources' / 'Photos' / 'me3.jpg'
OUT_DIR = ROOT / 'Outputs' / 'Section1_Section2_me3'
OUT_DIR.mkdir(parents=True, exist_ok=True)

img = cv.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f'Could not read image: {IMAGE_PATH}')


def save(name, image):
    path = OUT_DIR / name
    cv.imwrite(str(path), image)
    print(f'Saved: {path}')

# Section 1 - Basics
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img, (7, 7), 0)
canny = cv.Canny(gray, 100, 200)
resized = cv.resize(img, None, fx=0.6, fy=0.6, interpolation=cv.INTER_AREA)
# simple crop
cropped = img[80:420, 120:520]
# translation
rows, cols = img.shape[:2]
M = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv.warpAffine(img, M, (cols, rows))
# rotation
M_rot = cv.getRotationMatrix2D((cols // 2, rows // 2), 25, 1.0)
rotated = cv.warpAffine(img, M_rot, (cols, rows))

save('section1_gray.jpg', gray)
save('section1_blur.jpg', blur)
save('section1_canny.jpg', canny)
save('section1_resized.jpg', resized)
save('section1_cropped.jpg', cropped)
save('section1_translated.jpg', translated)
save('section1_rotated.jpg', rotated)

# Section 2 - Advanced
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

b, g, r = cv.split(img)
merged = cv.merge([b, g, r])

lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobelx = np.uint8(np.absolute(sobelx))
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
sobely = np.uint8(np.absolute(sobely))
combined_sobel = cv.bitwise_or(sobelx, sobely)

# Simple mask example
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv.rectangle(mask, (80, 80), (450, 350), 255, -1)
masked = cv.bitwise_and(img, img, mask=mask)

save('section2_hsv.jpg', hsv)
save('section2_lab.jpg', lab)
save('section2_channels_merged.jpg', merged)
save('section2_laplacian.jpg', lap)
save('section2_sobelx.jpg', sobelx)
save('section2_sobely.jpg', sobely)
save('section2_combined_sobel.jpg', combined_sobel)
save('section2_masked.jpg', masked)

print('Finished processing me3.jpg')
