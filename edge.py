import cv2
import numpy
from google.cloud import vision
import io
client = vision.ImageAnnotatorClient()

img = cv2.imread("data/crop_0.jpg")

#gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#canny = cv2.medianBlur(img, 3)

lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
gry = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
img = cv2.cvtColor(gry, cv2.COLOR_BGR2GRAY)

image = vision.Image(content=image)

response = client.text_detection(content=img)
texts = response.text_annotations
print('Texts:')

for text in texts:
    print('\n"{}"'.format(text.description))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
                for vertex in text.bounding_poly.vertices])

    print('bounds: {}'.format(','.join(vertices)))

if response.error.message:
    raise Exception(
        '{}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors'.format(
            response.error.message))