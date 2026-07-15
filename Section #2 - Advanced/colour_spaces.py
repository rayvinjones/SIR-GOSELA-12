#pylint:disable=no-member

from pathlib import Path
import cv2 as cv

root = Path(__file__).resolve().parent.parent
img_path = root / 'Resources' / 'Photos' / 'diether1.jpg'


def show(name, image, max_width=900, max_height=700):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    width = max(1, int(w * scale))
    height = max(1, int(h * scale))
    resized = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, width, height)
    cv.imshow(name, resized)


img = cv.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f'Could not read image: {img_path}')

show('me3.jpg', img)

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
show('Gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
show('HSV', hsv)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
show('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
show('RGB', rgb)

# HSV to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
show('LAB --> BGR', lab_bgr)

cv.waitKey(0)
cv.destroyAllWindows()