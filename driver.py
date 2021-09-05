from mainOCR import OCR
import cv2


img = cv2.imread("data/Image1.jpg")
isOnline = False

headstone = ""
headstone = OCR(img, isOnline)

print(headstone)