    #pylint:disable=no-member

from pathlib import Path
import cv2 as cv

# Read in an image using a path that works from any current working directory
root = Path(__file__).resolve().parent.parent
img_path = root / 'Resources' / 'Photos' / 'diether1.jpg'
img = cv.imread(str(img_path))

if img is None:
    raise FileNotFoundError(f'Could not read image: {img_path}')

def show(name, image, width=700):
    h, w = image.shape[:2]
    scale = width / w
    height = int(h * scale)
    resized = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    cv.imshow(name, resized)

show('me3.jpg', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
show('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
show('Blur', blur)

# Edge Cascade
canny = cv.Canny(blur, 125, 175)
show('Canny Edges', canny)

# Dilating the image
dilated = cv.dilate(canny, (7, 7), iterations=3)
show('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (7, 7), iterations=3)
show('Eroded', eroded)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
show('Resized', resized, width=500)

# Cropping
cropped = img[50:200, 200:400]
show('Cropped', cropped, width=500)

cv.waitKey(0)
cv.destroyAllWindows()