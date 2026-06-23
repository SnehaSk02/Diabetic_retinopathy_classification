import cv2
import numpy as np

def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)[1]

    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    return img


def apply_clahe(img):

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    merged = cv2.merge((l,a,b))

    enhanced= cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2RGB
    )
    return enhanced